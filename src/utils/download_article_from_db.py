#!/usr/bin/python3
#-*- coding: utf-8 -*-
#sudo pip3 install mysql-connector
#

import sys
import os
import mysql.connector

DB_USER = 'root'
DB_NAME = 'news'
DB_HOST = '47.95.229.7'
DB_PORT = 13306

def _save_content(file_dir, file_name, content):
    '''
    保存文件
    '''
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    file_path = file_dir + '/' + file_name + '.txt'
    
    if os.path.exists(file_path):
        print('{file_name} already exist !'.format(file_name = file_name))
        return
    
    with open(file_path, 'a') as out:
        out.write(content)
        print('save content to ./{file_dir}/{file_name}.'.format(file_dir = file_dir, file_name = file_name))

def main(password, categoryId, limit = 10000, user = DB_USER, database = DB_NAME, host = DB_HOST, port = DB_PORT):
    print('querying...')
    
    query = ("SELECT id, content FROM articles WHERE category_id = " + categoryId + " LIMIT " + str(limit))
    
    cnx = mysql.connector.connect(**{
        'user': user,
        'password': password,
        'host': host,
        'database': database,
        'port': port
    })
    cur = cnx.cursor(buffered=True)
    cur.execute(query, (categoryId))

    for (id, content) in cur:
        _save_content(categoryId, id, content)
        pass

    cnx.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ./download_article_from_db.py password categoryId(1 - 10) [limit = 10000]')
        quit()

    limit = 10000
    if len(sys.argv) is 4:
        limit = int(sys.argv[3])
        
    main(sys.argv[1], sys.argv[2], limit)
