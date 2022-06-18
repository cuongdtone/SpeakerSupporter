# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import sqlite3
from annoy import AnnoyIndex
from pathlib import Path

QUERY_INSERT_EMP = "INSERT INTO user ( id, fullname, sex, vocative, position, face_feature) VALUES ( ?, ?, ?, ?, ?, ?);"
QUERY_UPDATE_EMP = "UPDATE employee SET fullname = ?, sex = ?, vocative = ?, updated_user = ?, isadmin=?, status = ?, active = ? WHERE code = ?;"
QUERY_UPDATE_STATUS_EMP = "UPDATE employee SET status = ? WHERE code = ?;"
QUERY_INSERT_TKP = """INSERT INTO timekeepings ( id, fullname, position, start_time, end_time, followers) VALUES (?, ?, ?, ?, ?, ?);"""
QUERY_UPDATE_FEATURE_EMP = "UPDATE employee SET face_feature = ? WHERE code = ?;"



def connect_database():
    conn = sqlite3.connect('./src/SpeakerDB.db')
    return conn


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def execute(db, query, values=None):
    try:
        if db == None:
            db = connect_database()

        cur = db.cursor()
        cur.execute(query, values)
        db.commit()

        cur.close()
        return True
    except Exception as ex:
        db.rollback()
        print(ex, '--------------------')
        return False


def get_employee_code(db=None):
    if db == None:
        db = connect_database()
    cur = db.cursor()
    result = cur.execute('SELECT MAX(code) from employee;')
    result_data = result.fetchone()
    cur.close()
    if result == None or len(result_data) > 0:
        return str(int(result_data[0]) + 1)
    else:
        return "10001"


def insert_speaker(db, id: str, fullname: str, sex: int, position: str, face_feature: str):
    if sex == 0:
        vocative = "Anh|Bạn"
    elif sex == 1:
        vocative = "Chị|Bạn"
    else:
        vocative = "Bạn"

    return execute(db, QUERY_INSERT_EMP, (id, fullname, sex, vocative, position, face_feature))


def update_employee(db, code: str, fullname: str, sex: int, updated_user: str, isadmin=False, status=0, active=False):
    if sex == 0:
        vocative = "Anh|Bạn"
    elif sex == 1:
        vocative = "Chị|Bạn"
    else:
        vocative = "Bạn"

    return execute(db, QUERY_UPDATE_EMP, (fullname, sex, vocative, updated_user, isadmin, status, active, code))

def update_face_feature_employee(db, code: str, feat: str):
    return execute(db, QUERY_UPDATE_FEATURE_EMP, (feat, code))

def update_status_employee(db, code: str, status=1):
    return execute(db, QUERY_UPDATE_STATUS_EMP, (status, code))


def get_all_employee(db):
    try:
        query = "SELECT id, fullname, sex, vocative, position, face_feature FROM user;"
        if db is None:
            db = connect_database()
        cur = db.cursor()
        result = cur.execute(query)
        result_data = result.fetchall()
        employee = []
        for row in result_data:
            employee.append(dict_factory(cur, row))

        cur.close()
        return list(employee)
    except Exception as ex:
        return None

def get_all_timekeeping(db=None):
    if db == None:
        db = connect_database()
    try:
        query = "SELECT id, fullname, position,start_time, end_time, followers FROM timekeepings;"
        if db is None:
            db = connect_database()
        cur = db.cursor()
        result = cur.execute(query)
        result_data = result.fetchall()
        employee = []
        for row in result_data:
            employee.append(dict_factory(cur, row))

        cur.close()
        return list(employee)
    except Exception as ex:
        return None

def insert_timekeeping(code: str, fullname: str,  postion: str, start_time: str, end_time:str, followers: str):
    db = connect_database()
    return execute(db, QUERY_INSERT_TKP, (code, fullname,  postion, start_time, end_time, followers))


def insert_casia(code: str, embed: str):
    db = connect_database()
    QUERY = """INSERT INTO test_casia ( id, embed) VALUES (?, ?);"""
    return execute(db, QUERY, (code, embed))


def get_casia(db):
    try:
        query = "SELECT id, embed FROM test_casia;"
        if db is None:
            db = connect_database()
        cur = db.cursor()
        result = cur.execute(query)
        result_data = result.fetchall()
        employee = []
        for row in result_data:
            employee.append(dict_factory(cur, row))

        cur.close()
        return list(employee)
    except Exception as ex:
        return None


root_path = 'src/data/'

def load_user_data():
    users_info = get_all_employee(None)
    tree = AnnoyIndex(512, 'euclidean')
    data = []
    count = 0
    for user in users_info:
        if user["face_feature"] is not None and len(user["face_feature"]) != 0:
            info = [user['id'], user['fullname'], user['position']]
            feat = str2list(user['face_feature'])
            tree.add_item(count, feat)
            data.append(info)
            count +=1
    tree.build(100)
    data = np.array(data)
    print(data)
    return data, tree, users_info


def str2list(face_feature):
    str = face_feature.strip(']')[1:]
    return [float(i.strip('\n')) for i in str.split()]

import numpy as np


def str2np(face_feature):
    str = face_feature.strip('[')
    str = str.strip(']')
    float_lst = [float(i.strip('\n')) for i in str.split()]
    return np.array(float_lst, dtype=np.float)