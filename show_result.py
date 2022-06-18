# -*- coding: utf-8 -*-
# @Organization  : DUT
# @Author        : Cuong Tran, Yen Le
# @Time          : 2022-06-18

import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3 as sqlite

def get_all_employee(name_table='employees'):
    con = sqlite.connect("src/SpeakerDB.db")
    with con:
        cur = con.cursor()
        cur.execute("SELECT id, fullname, position,start_time, end_time, followers FROM timekeepings;")
        # get data
        rows = cur.fetchall()
        data = []
        for id, row in enumerate(rows):
            row = list(row)
            for i in [len(row)-1]:
                # convert binary to array
                arr = np.frombuffer(row[i], dtype='int')
                # convert array 1D to nD
                row[i] = arr.reshape(int(len(arr)/2), 2)
            data.append(row)
        return data

data = get_all_employee('timekeepings')
print(data)

for i in data[::-1]:
    id, name, position, from_time, to_time, followers = i
    fig = plt.figure()
    # print(followers.shape)
    plt.plot(range(followers.shape[0]), followers[:, 0])
    plt.plot(range(followers.shape[0]), followers[:, 1])
    plt.title(f'{position.title()} {name} from {from_time} to {to_time}')
    plt.xlabel('Time')
    plt.ylabel('Number person')
    fig.legend(['Concentrate people', "Not concentrate people"])
    plt.show()
