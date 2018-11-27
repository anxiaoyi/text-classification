#-* -coding:utf-8 -*-
#
import os
import sys

def _save_file(file_path, content):
    with open(file_path, 'w') as out:
        out.write(content)

def _process_sina(content):
    try:
        bIndex = content.index('新浪娱乐讯')
        content = content[bIndex:]
    except:
        ValueError

    if content.startswith('window.sina'):
        try:
            eIndex = content.index('}')
            content = content[eIndex + 1:]
        except:
            ValueError

    try:
        funcIndex = content.index('(function')
        content = content[:funcIndex]
    except:
        ValueError

    return content

def _process_line(file_path):
    lines = open(file_path, 'r').read().split('\n')
    finalLines = []
    
    for line in lines:
        if '#endText' in line \
           or '.ct_hqimg' in line \
           or 'if (/(iPhone|iPad' in line \
           or 'window.NTES' in line \
           or 'BAIDU_CLB_SLOT_ID' in line \
           or 'var cpro_id' in line \
           or '/*300*250' in line \
           or line.strip() == '':
            continue

        finalLines.append(line)

    return '\n'.join(finalLines)

def process(file_path):
    '''
    pre processor single file
    '''
    content = _process_line(file_path)
    content = content.strip()

    # ==============================
    # 乱码
    # ==============================

    if '�' in content:
        c = content.count('�')
        
        if c >= 15:
            return None
        else:
            content = content.replace('�', '')

    content = _process_sina(content)

    return content.strip()

def pre_processor_category(category_id):
    directory = str(category_id)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            content = process(file_path)
            if content is None:
                os.remove(file_path)
                print('[WARN] Delete file cause of too many � in ' + file_path)
            else:
                _save_file(file_path, content)
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 text_pre_processor.py categoryId(1 - 10)')
        quit()

    categoryId = sys.argv[1]
    pre_processor_category(categoryId)
