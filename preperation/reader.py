#!/usr/bin/python

##
## HOW TO RUN: python <path-to-this-file> ./
import os
import sys
import gzip
import zlib
import json
import bz2
import tempfile
import requests
import subprocess32 as subprocess
from aenum import Enum
import matplotlib.pyplot as plt
import numpy as np
import scipy
import platform
import cv2
if platform.system() == "Darwin":
  os.environ["LA_LIBRARY_FILEPATH"] = "/usr/local/opt/libarchive/lib/libarchive.dylib"
import libarchive.public

from cereal import log as capnp_log

OP_PATH = os.path.dirname(os.path.dirname(capnp_log.__file__))
class DataUnreadableError(Exception):
  pass
def FileReader(fn):
  return open(fn, 'rb')

def convert_old_pkt_to_new(old_pkt):
  m, d = old_pkt
  msg = capnp_log.Event.new_message()

  if len(m) == 3:
    _, pid, t = m
    msg.logMonoTime = t
  else:
    t, pid = m
    msg.logMonoTime = int(t * 1e9)

  last_velodyne_time = None

  if pid == PID_OBD:
    write_can_to_msg(d, 0, msg)
  elif pid == PID_CAM:
    frame = msg.init('frame')
    frame.frameId = d[0]
    frame.timestampEof = msg.logMonoTime
  # iOS
  elif pid == PID_IGPS:
    loc = msg.init('gpsLocation')
    loc.latitude = d[0]
    loc.longitude = d[1]
    loc.speed = d[2]
    loc.timestamp = int(m[0]*1000.0)   # on iOS, first number is wall time in seconds
    loc.flags = 1 | 4  # has latitude, longitude, and speed.
  elif pid == PID_IMOTION:
    user_acceleration = d[:3]
    gravity = d[3:6]

    # iOS separates gravity from linear acceleration, so we recombine them.
    # Apple appears to use this constant for the conversion.
    g = -9.8
    acceleration = [g*(a + b) for a, b in zip(user_acceleration, gravity)]

    accel_event = msg.init('sensorEvents', 1)[0]
    accel_event.acceleration.v = acceleration
  # android
  elif pid == PID_GPS:
    if len(d) <= 6 or d[-1] == "gps":
      loc = msg.init('gpsLocation')
      loc.latitude = d[0]
      loc.longitude = d[1]
      loc.speed = d[2]
      if len(d) > 6:
        loc.timestamp = d[6]
      loc.flags = 1 | 4  # has latitude, longitude, and speed.
  elif pid == PID_ACCEL:
    val = d[2] if type(d[2]) != type(0.0) else d
    accel_event = msg.init('sensorEvents', 1)[0]
    accel_event.acceleration.v = val
  elif pid == PID_GYRO:
    val = d[2] if type(d[2]) != type(0.0) else d
    gyro_event = msg.init('sensorEvents', 1)[0]
    gyro_event.init('gyro').v = val
  elif pid == PID_LIDAR:
    lid = msg.init('lidarPts')
    lid.idx = d[3]
  elif pid == PID_APPLANIX:
    loc = msg.init('liveLocation')
    loc.status = d[18]

    loc.lat, loc.lon, loc.alt = d[0:3]
    loc.vNED = d[3:6]

    loc.roll = d[6]
    loc.pitch = d[7]
    loc.heading = d[8]

    loc.wanderAngle = d[9]
    loc.trackAngle = d[10]

    loc.speed = d[11]

    loc.gyro = d[12:15]
    loc.accel = d[15:18]
  elif pid == PID_IBAROMETER:
    pressure_event = msg.init('sensorEvents', 1)[0]
    _, pressure = d[0:2]
    pressure_event.init('pressure').v = [pressure] # Kilopascals
  elif pid == PID_IINIT and len(d) == 4:
    init_event = msg.init('initData')
    init_event.deviceType = capnp_log.InitData.DeviceType.chffrIos

    build_info = init_event.init('iosBuildInfo')
    build_info.appVersion = d[0]
    build_info.appBuild = int(d[1])
    build_info.osVersion = d[2]
    build_info.deviceModel = d[3]

  return msg.as_reader()


def index_log(fn):
  index_log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "index_log")
  index_log = os.path.join(index_log_dir, "index_log")
  phonelibs_dir = os.path.join(OP_PATH, 'phonelibs')

  subprocess.check_call(["make", "PHONELIBS=" + phonelibs_dir], cwd=index_log_dir, stdout=open("/dev/null","w"))

  try:
    dat = subprocess.check_output([index_log, fn, "-"])
  except subprocess.CalledProcessError:
    raise DataUnreadableError("%s capnp is corrupted/truncated" % fn)
  return np.frombuffer(dat, dtype=np.uint64)

def event_read_multiple_bytes(dat):
  with tempfile.NamedTemporaryFile() as dat_f:
    dat_f.write(dat)
    dat_f.flush()
    idx = index_log(dat_f.name)

  end_idx = np.uint64(len(dat))
  idx = np.append(idx, end_idx)

  return [capnp_log.Event.from_bytes(dat[idx[i]:idx[i+1]])
          for i in xrange(len(idx)-1)]


# this is an iterator itself, and uses private variables from LogReader
class MultiLogIterator(object):
  def __init__(self, log_paths, wraparound=True):
    self._log_paths = log_paths
    self._wraparound = wraparound

    self._first_log_idx = next(i for i in xrange(len(log_paths)) if log_paths[i] is not None)
    self._current_log = self._first_log_idx
    self._idx = 0
    self._log_readers = [None]*len(log_paths)
    self.start_time = self._log_reader(self._first_log_idx)._ts[0]

  def _log_reader(self, i):
    if self._log_readers[i] is None and self._log_paths[i] is not None:
      log_path = self._log_paths[i]
      print "LogReader:", log_path
      self._log_readers[i] = LogReader(log_path)

    return self._log_readers[i]

  def __iter__(self):
    return self

  def _inc(self):
    lr = self._log_reader(self._current_log)
    if self._idx < len(lr._ents)-1:
      self._idx += 1
    else:
      self._idx = 0
      self._current_log = next(i for i in xrange(self._current_log + 1, len(self._log_readers) + 1) if i == len(self._log_readers) or self._log_paths[i] is not None)
      # wraparound
      if self._current_log == len(self._log_readers):
        if self._wraparound:
          self._current_log = self._first_log_idx
        else:
          raise StopIteration

  def next(self):
    while 1:
      lr = self._log_reader(self._current_log)
      ret = lr._ents[self._idx]
      if lr._do_conversion:
        ret = convert_old_pkt_to_new(ret, lr.data_version)
      self._inc()
      return ret

  def tell(self):
    # returns seconds from start of log
    return (self._log_reader(self._current_log)._ts[self._idx] - self.start_time) * 1e-9

  def seek(self, ts):
    # seek to nearest minute
    minute = int(ts/60)
    if minute >= len(self._log_paths) or self._log_paths[minute] is None:
      return False

    self._current_log = minute

    # HACK: O(n) seek afterward
    self._idx = 0
    while self.tell() < ts:
      self._inc()
    return True


class LogReader(object):
  def __init__(self, fn, canonicalize=True):
    _, ext = os.path.splitext(fn)
    data_version = None

    with FileReader(fn) as f:
      dat = f.read()

    # decompress file
    if ext == ".gz" and ("log_" in fn or "log2" in fn):
      dat = zlib.decompress(dat, zlib.MAX_WBITS|32)
    elif ext == ".bz2":
      dat = bz2.decompress(dat)
    elif ext == ".7z":
      with libarchive.public.memory_reader(dat) as aa:
        mdat = []
        for it in aa:
          for bb in it.get_blocks():
            mdat.append(bb)
      dat = ''.join(mdat)

    # TODO: extension shouln't be a proxy for DeviceType
    if ext == "":
      if dat[0] == "[":
        needs_conversion = True
        ents = [json.loads(x) for x in dat.strip().split("\n")[:-1]]
        if "_" in fn:
          data_version = fn.split("_")[1]
      else:
        # old rlogs weren't bz2 compressed
        needs_conversion = False
        ents = event_read_multiple_bytes(dat)
    elif ext == ".gz":
      if "log_" in fn:
        # Zero data file.
        ents = [json.loads(x) for x in dat.strip().split("\n")[:-1]]
        needs_conversion = True
      elif "log2" in fn:
        needs_conversion = False
        ents = event_read_multiple_bytes(dat)
      else:
        raise Exception("unknown extension")
    elif ext == ".bz2":
      needs_conversion = False
      ents = event_read_multiple_bytes(dat)
    elif ext == ".7z":
      needs_conversion = True
      ents = [json.loads(x) for x in dat.strip().split("\n")]
    else:
      raise Exception("unknown extension")

    if needs_conversion:
      # TODO: should we call convert_old_pkt_to_new to generate this?
      self._ts = [x[0][0]*1e9 for x in ents]
    else:
      self._ts = [x.logMonoTime for x in ents]

    self.data_version = data_version
    self._do_conversion = needs_conversion and canonicalize
    self._ents = ents

  def __iter__(self):
    for ent in self._ents:
      if self._do_conversion:
        yield convert_old_pkt_to_new(ent, self.data_version)
      else:
        yield ent

def load_many_logs_canonical(log_paths):
  """Load all logs for a sequence of log paths."""
  for log_path in log_paths:
    for msg in LogReader(log_path):
      yield msg
def reader(path, num,outpath):
    
    print(path)
    steering_angles = [] # Empty list 
    features = []
    gas = []
    brake = []
    
    log_path = path + "/rlog.bz2"
    video_path = path+ "/fcamera.hevc"
    if os.path.exists(log_path) is True and os.path.exists(video_path) is True:
        print(log_path)
        lr = LogReader(log_path)
        logs = list(lr)
        for l in logs:
            if l.which() == "carControl":
                steering_angles.append(round(float(l.carControl.actuators.steerAngle),2))
                gas.append(round(float(l.carControl.actuators.gas),2))
                brake.append(round(float(l.carControl.actuators.brake),2))
                #print(round(float(l.carControl.actuators.gas),2))
                #print(round(float(l.carControl.actuators.brake),2))
        print("Finished labels")
        print(video_path)
        
        vidObj = cv2.VideoCapture(video_path) 
        count = 0
        su = 1
    
        while su == 1: 
            success, image = vidObj.read() 
            if success:
                processed_image = cv2.resize(image, (640,480))
                features.append(processed_image)
                #print("Image appended")
            else: 
                su = 0
        
    print("Saved video and label and processed " + str(num))
    np.save(outpath+"/camera"+str(num)+".npy", np.array(features))
    steering_angles = np.array(steering_angles)
    gas = np.array(gas)
    brake = np.array(brake)
    labels = np.stack((steering_angles, gas,brake), axis=1)
    #print(labels.shape, labels[::100])
    np.save(outpath+"/labels"+str(num)+".npy", labels)

if __name__ == "__main__":
    count = 1
    pathtofiles = sys.argv[1]
    outputpath = sys.argv[2]
    os.path.join(pathtofiles)
    for i in os.listdir(pathtofiles):
        print(i)
        reader(i, count,outputpath)
        count = count +1
    
