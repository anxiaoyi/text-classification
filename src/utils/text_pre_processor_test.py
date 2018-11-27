#-* -coding:utf-8 -*-
#

from text_pre_processor import process

def t(file_name):
    content = process(file_name)
    
    print('file: ' + file_name + ':\n')
    print(content)
    print('=====================\n\n')

t('1/000e614c33239863d75715b0338f4cad.txt')
# t('1/0012df09751baadf79ec9c96109632d2.txt')
# t('1/001521ca3c91764749d0765853432958.txt')
# t('1/00640cc00593940aa07efea13628faca.txt')
# t('1/163ea28e8f36e00bbc93b0076be33446.txt')
# t('1/14fe85c04516f1656c913fc4f8d0ff70.txt')
# t('1/00780476473714317611af60151da047.txt')
# t('1/152d716eb42c2125666efcd4d7cda453.txt')
# t('1/1435ad4ac48e4ce1f8d1a8a75d531872.txt')

# t('2/000197e457967ebfd211f6c22ee3c16e.txt')
# t('2/06c607fd7f9d0d384dc46bd778b51295.txt')
# t('2/06922561ad46f7093a6461209ac49f07.txt')

# t('3/037e46bc2e7395e7ebc3d1da9b155183.txt')
# t('4/00a46fa9bd0080035058d50a4b31664b.txt')
# t('5/008abd92eefd1e7491d05f0df0cfa26c.txt')
