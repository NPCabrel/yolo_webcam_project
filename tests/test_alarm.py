import pytest
import numpy as np
from src.alarm_system import DistanceAlarm
from src.config import AlarmConfig

def test_alarm_initialization():
    alarm = DistanceAlarm()
    assert alarm.alarm_counter == 0
    assert alarm.total_frames_alarm == 0

def test_alarm_no_persons():
    alarm = DistanceAlarm(AlarmConfig(min_distance_px=150))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    triggered, _, centers = alarm.check_alarm([], frame)
    assert triggered == False
    assert len(centers) == 0

def test_alarm_two_persons_far():
    alarm = DistanceAlarm(AlarmConfig(min_distance_px=150))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = [
        {"center": (100, 200)},
        {"center": (400, 200)}
    ]
    triggered, _, centers = alarm.check_alarm(persons, frame)
    assert triggered == False
    assert len(centers) == 2

def test_alarm_two_persons_close():
    alarm = DistanceAlarm(AlarmConfig(min_distance_px=150))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = [
        {"center": (100, 200)},
        {"center": (120, 200)}
    ]
    triggered, _, centers = alarm.check_alarm(persons, frame)
    assert triggered == True
    assert len(centers) == 2