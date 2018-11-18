# -*- coding: utf-8 -*-
from qntstock.local import DB_PATH, PATH, FS_PATH, FS_PATH_OL
from pymongo import MongoClient, UpdateOne
import os
import sys
from sqlalchemy import create_engine
from logger import LogTxt
import datetime
import time
import traceback
import pandas as pd

class Utils(object):
    def __init__(self, log_dir='default_log'):
        self.logger = LogTxt(log_dir).logger

    def get_conn(self):
        return MongoClient('mongodb://127.0.0.1:27017')
        
    def update_doc(self, docs, db_name, coll_name):
        if isinstance(docs, dict):
            docs = [docs, ]
        client = self.get_conn()
        db = client[db_name]
        coll = db[coll_name]
        update_requests = []
        for doc in docs:
            update_requests.append(
                UpdateOne(
                    {'code': doc['code'], 'date': doc['date']},
                    {'$set': doc},
                    upsert=True)
            )
        if len(update_requests) > 0:
            update_result = coll.bulk_write(update_requests, ordered=False)
            # self.logger.info('保存日线数据，代码： %s, 插入：%4d条, 更新：%4d条' %
            #         (code, update_result.upserted_count, update_result.modified_count))
        client.close()

    def update_kdata_df(self, code, df, db_name, coll_name):
        client = self.get_conn()
        db = client[db_name]
        coll = db[coll_name]
        update_requests = []
        for df_index in df.index:
            doc = dict(df.loc[df_index])
            doc['code'] = code
            update_requests.append(
                UpdateOne(
                    {'code': doc['code'], 'date': doc['date']},
                    {'$set': doc},
                    upsert=True)
            )
        if len(update_requests) > 0:
            update_result = coll.bulk_write(update_requests, ordered=False)
            # self.logger.info('保存日线数据，代码： %s, 插入：%4d条, 更新：%4d条' %
            #         (code, update_result.upserted_count, update_result.modified_count))
        client.close()

    def update_daily_df(self, date, df, db_name, coll_name):
        client = self.get_conn()
        db = client[db_name]
        coll = db[coll_name]
        update_requests = []
        for df_index in df.index:
            doc = dict(df.loc[df_index])
            doc['date'] = date
            update_requests.append(
                UpdateOne(
                    {'code': doc['code'], 'date': doc['date']},
                    {'$set': doc},
                    upsert=True)
            )
        if len(update_requests) > 0:
            update_result = coll.bulk_write(update_requests, ordered=False)
            # self.logger.info('保存基本数据，日期： %s, 插入：%4d条, 更新：%4d条' %
            #         (date, update_result.upserted_count, update_result.modified_count))
        client.close()

    def get_all_code_list(self, start_date, end_date):
        client = self.get_conn()
        coll = client['basic']['dayall']
        docs = coll.distinct('code', {'date': {'$lte': end_date, '$gte': start_date}})
        return docs
        
    def find_code_date(self, db_name, coll_name, code, start_date, end_date):
        client = self.get_conn()
        db = client[db_name]
        coll = db[coll_name]
        query = {'code': code, 'date': {'$lte': end_date, '$gte': start_date}}
        docs = coll.find(query)
        s = {}
        for r in docs:
            date = r['date']
            s[date] = r
        client.close()
        return s

    def unixtime2str(self, ut):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ut))

    def str2unixtime(self, s):
        return time.mktime(time.strptime(s, '%Y-%m-%d %H:%M:%S'))

    def json_unicode_to_string(self, data):
        if isinstance(data, list):
            return self._decode_list(data)
        elif isinstance(data, dict):
            return self._decode_dict(data)
        else:
            return None

    def _decode_list(self, data):
        rv = []
        for item in data:
            if isinstance(item, list):
                item = self._decode_list(item)
            elif isinstance(item, dict):
                item = self._decode_dict(item)
            rv.append(item)
        return rv

    def _decode_dict(self, data):
        rv = {}
        for key, value in data.iteritems():
            if isinstance(value, list):
                value = self._decode_list(value)
            elif isinstance(value, dict):
                value = self._decode_dict(value)
            rv[key] = value
        return rv

   #date
    def get_logdate(self, basedate='today', offset=None, fmt='%Y-%m-%d'):
        assert type(offset) == int or offset is None, 'TypeError: the type of parameter offset should be int'
        offset = 0 if offset is None else offset
        if basedate == 'today':
            rdate = (datetime.date.today() + datetime.timedelta(days = offset)).strftime(fmt)
        else:
            rdate = (datetime.datetime.strptime(basedate,fmt) + datetime.timedelta(days = offset)).strftime(fmt)
        return rdate

    def get_date_diff(self, start_logdate, end_logdate, fmt='%Y-%m-%d'):
        start_date = datetime.datetime.strptime("%s"%start_logdate, fmt).date()
        end_date = datetime.datetime.strptime("%s"%end_logdate, fmt).date()
        date_delta = (end_date - start_date).days
        return date_delta

    def get_last_date(self, fmt='%Y-%m-%d'):
        return (datetime.date.today() - datetime.timedelta(days = 1)).strftime(fmt)

    def get_date_list(self, start_date, end_date):
        cur_date = start_date
        date_list = []
        date_diff = self.get_date_diff(start_date, end_date)
        for _ in range(date_diff+1):
            date_list.append(cur_date)
            cur_date = self.get_logdate(cur_date, 1)
        return date_list


class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s=None):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = round(self.width * self.count / self.total)
        sys.stdout.write('{0:4} {1:4}/{2:4}: '.format(s, self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


def _sort(sdf):
    tdf = sdf.sort_values(by='date')
    tdf = tdf.reset_index(drop=True)
    return tdf

if __name__ == "__main__":
    u=Utils('test_log')
    ld = '20180101'
    for i in range(2):
        u.logger.info('This is info message')
        u.logger.debug('This is debug message')
        print(u.get_logdate(ld, i))
