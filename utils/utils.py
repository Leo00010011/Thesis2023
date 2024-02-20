import cv2
import datetime
import json
import os
import glob
from pathlib import Path
from tools.bag_utils import RSRecord, RSPlayback


def parse_date(s: str):
    date, time = tuple(s.split(' '))
    year, month, day = tuple(date.split('-'))
    hour, minutes, seconds = tuple(time.split(':'))
    result = {
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'hour': int(hour),
        'minutes': int(minutes),
        'seconds': int(seconds)
    }
    return result


'2023-09-10 07:16:30'

def number_str(n,size):
    s = str(n)[:size]
    return "0"*(size - len(s)) + s


def now_str():
    return str(datetime.datetime.now())[:19]

def date_str(date):
    date_s = '-'.join([number_str(date['year'],4),number_str(date['month'],2),number_str(date['day'],2)])
    time_s = ':'.join([number_str(date['hour'],2),number_str(date['minutes'],2),number_str(date['seconds'],2)])
    return ' '.join([date_s,time_s])

def process_video_info(data_path):
    sub_folders = glob.glob(f'{data_path}/*/')
    for folder in sub_folders:
        #cargar el json
        json_path = os.path.join(folder,'data.json')
        patient_data = None
        with open(json_path,'r') as f:
            patient_data = json.load(f)
        #Añadir los videos
        for bag_path in glob.glob(f'{folder}/*.bag'):
            name = Path(bag_path).name.split('.')[0]
            video_dict = {
                'path': bag_path,
                'date':{
                    'year': int(name[0:4]),
                    'month': int(name[4:6]),
                    'day': int(name[6:8]),
                    'hour': int(name[9:11]),
                    'minutes': int(name[11:13]),
                    'seconds': int(name[13:15])
                }
            }
            patient_data['Videos'].append(video_dict)
        with open(json_path,'w') as f:
            json.dump(patient_data,f)


def fecha_dict_to_str(fecha_dict):
    date = str(datetime.datetime(
        fecha_dict['year'], fecha_dict['month'], fecha_dict['day']))[:10]
    return date

def video_name(date):
    s_date = ''.join([number_str(date['year'],4), number_str(date['month'],2), number_str(date['day'],2)])
    s_time = ''.join([number_str(date['hour'],2),number_str(date['minutes'],2),number_str(date['seconds'],2)])
    return '-'.join([s_date,s_time])


def get_patient_folder_name(patient):
        nombre = patient['Name']
        fecha = fecha_dict_to_str(patient['First_Date'])
        return f'{fecha}-{nombre}'

def record(save_path):
    reader = RSRecord(save_path)
    reader.start_camera()
    for color in reader.get_color_frames():
        color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        cv2.putText(color, 'Presiona \'s\' para terminar', (0, 30), 0, .75, (0,255,0), thickness=2)
        cv2.imshow('Prueba',color)
        key = cv2.waitKey(10)
        if key == 115:
            break
    cv2.destroyAllWindows()
    reader.stop_camera()

# def view_camera_stream():
#     reader = RSReader()
#     reader.start_camera()
#     for color in reader.get_color_frames():
#         color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
#         cv2.imshow('Prueba',color)
#         key = cv2.waitKey(int(1000/60))
#         if key == 115:
#             break
#     reader.stop_camera()

def view_playback(data_path):
    reader = RSPlayback(data_path)
    reader.start_camera()
    for color, depth in reader.get_frames():
        color = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
        cv2.putText(color, 'Presiona \'s\' para terminar', (0, 30), 0, .75, (0,255,0), thickness=2)
        cv2.imshow('Prueba',color)
        key = cv2.waitKey(int(1000/60))
        if key == 115:
            break
    reader.stop_camera()
    cv2.destroyAllWindows()
