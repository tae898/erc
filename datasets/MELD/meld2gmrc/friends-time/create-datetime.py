import csv
import os
from glob import glob
from dateutil import parser
import json
import pickle
from datetime import datetime, timedelta

friends_time_ = sorted(glob('./*.csv'))
friends_time = {}


foo = []
for path in list(friends_time_):
    seasonno = path.split('/')[-1].split('.')[0].split('-')[-1]
    friends_time[seasonno] = {}

    with open(path) as f:
        reader = csv.reader(f)
        parsed = list(reader)
    parsed = parsed[1:]

    for p in parsed:
        if seasonno == 'special':
            episodeno = p[0]
            date = p[-2]
        else:
            episodeno = p[1]
            date = p[-3]

        foo.append(episodeno)

        date = date.replace("\xa0", " ")
        date = date.replace("[b]", "")
        date = parser.parse(date)

        # Friends was aired at 20:00
        if (len(episodeno) == 4) or (len(episodeno) == 6):
            episodeno_1 = episodeno[:len(episodeno)//2]
            episodeno_2 = episodeno[len(episodeno)//2:]
            friends_time[seasonno][episodeno_1] = date + timedelta(hours=20)
            friends_time[seasonno][episodeno_2] = date + \
                timedelta(days=7) + timedelta(hours=20)

        else:
            friends_time[seasonno][episodeno] = date + timedelta(hours=20)


with open('friends-time.pkl', 'wb') as stream:
    pickle.dump(friends_time, stream)
