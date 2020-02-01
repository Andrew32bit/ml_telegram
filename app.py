from datetime import datetime
from nltk import *
import nltk
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict,Counter
import re
import time
import numpy as np
import math
import html2text
import gensim
import warnings
import requests
import pandas as pd
import telebot
import os
import yaml
warnings.filterwarnings("ignore")


with open('data.yaml', 'r') as f:
    doc = yaml.load(f)
telegram_token = doc["treeroot"]["telegram_token"]
token = doc["treeroot"]["vk_token"]
bot = telebot.TeleBot(telegram_token)
w2v_fpath = "/Users/andreas/PycharmProjects/med/ml_telegram/all.norm-sz100-w10-cb0-it1-min100.w2v"# min model #локально
#w2v_fpath="/app/all.norm-sz100-w10-cb0-it1-min100.w2v" #путь для докера
w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_fpath, binary=True, unicode_errors='ignore')
w2v.init_sims(replace=True)
df=pd.read_excel('/Users/andreas/PycharmProjects/med/ml_telegram/dict.xlsx') #локально
#df=pd.read_excel('/app/dict.xlsx') #путь для докера
dict_interests=df.to_dict('records')

text_file = open("/Users/andreas/PycharmProjects/med/ml_telegram/interests_3.txt", "r")#локально
#text_file = open("/app/interests_3.txt", "r") #путь для докера
interests_file = text_file.read().split(',')  # интересы лист
interests_file = [element.strip() for element in interests_file]  # удаляем пробелы
messages = {
    "start-message": "/help, чтобы посмотреть список команд",
    "help-message": '''/user_id - Определение и сравнение извлеченных интересов со словарями
                    \n/interests - Определение интересов пользователя'''
}


def convert_bbcode_fonts(html):
    flags = re.IGNORECASE | re.MULTILINE
    # replace start font tags
    html = re.sub(r'\[font\s*([^\]]+)\]', '<font \1>', html, flags=flags)
    # replace end font tags
    html = re.sub(r'\[/font\s*\]', '</font>', html, flags=flags)
    return html

def extract_text(html):
    html = convert_bbcode_fonts(html)
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    h.ignore_tables=True
    return h.handle(html)

def stem_words_array(words_array):
    stemmer = nltk.PorterStemmer()
    stemmed_words_array = []
    for word in words_array:
        if len(word)>3:
            try:
                stem = stemmer.stem(word)
                stemmed_words_array.append(stem)
            except Exception:
                pass

    return stemmed_words_array

def vectorize(X, model, y=None):
    word2weight = None
    dim = model.vector_size
    tfidf = TfidfVectorizer(lowercase=False, analyzer=lambda x: x)
    tfidf.fit(X)
    max_idf = max(tfidf.idf_)
    word2weight = defaultdict(
        lambda: max_idf,
        [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return np.array([
        np.mean([model[w] * word2weight[w]
                 for w in X if w in model] or
                [np.zeros(dim)], axis=0)
    ])

def create_cosine_simularity(query,w2v_vector):
    regex = re.compile("[А-Яа-я]+")
    text = regex.findall(query.lower())
    text = set(stem_words_array(text))
    v_query = vectorize(text,w2v)
    final_cos=[]
    for i in w2v_vector:
        cos_sim = cosine_similarity(i[0], v_query).flatten()[0]
        final_cos.append([cos_sim,i[1]])
    return final_cos

def extraction_interests(user_id):
    try:
        url='https://api.vk.com/method/groups.get?user_id={}&extended=1,fields=uid&v=5.52&access_token={}'.format(user_id,token)
        page = requests.get(url, verify=False, timeout=None)
        data = page.json()
        list_dict=[]
        for k,v in tqdm(data.items()):
            for x,y in v.items():
                list_dict.append(y)
        new_l=list_dict[1]
        if new_l=='User was deleted or banned':
            return 0
        final_list=[]
        for i in new_l:
            try:
                final_list.append(i.get('name'))
            except AttributeError:
                return 0
        text_from_groups=[i for i in final_list if i!='']
        return text_from_groups
    except Exception as e:
        return 0

# def extraction_interests(user_id):
#     extraction='verified, sex, bdate, home_town,online, domain, has_mobile, contacts, site, education,status, last_seen, followers_count, common_count, occupation, nickname, personal, connections, exports, activities, interests, music, movies, tv, books, games, about, quotes, can_post, can_see_all_posts, can_see_audio, can_write_private_message, can_send_friend_request, is_favorite, is_hidden_from_feed,screen_name, is_friend, friend_status, career'
#     url_another='https://api.vk.com/method/users.get?user_id={}&fields={},&v=5.52&access_token={}'.format(user_id,extraction,token) # запарсить интересы
#     #page= requests.get(url_another, verify=False, timeout=(1.0, 2))
#     page = requests.get(url_another, verify=False, timeout=None)
#     #time.sleep(3)
#     data = page.json()
#     inter=data.get('response')[0].get('interests')
#     games=data.get('response')[0].get('games')
#     music=data.get('response')[0].get('music')
#     movies=data.get('response')[0].get('movies')
#     act=data.get('response')[0].get('activities')
#     tv=data.get('response')[0].get('tv')
#     about=data.get('response')[0].get('about')
#     quotes=data.get('response')[0].get('quotes')
#     list_of_interests=[inter,act,tv,about,quotes,games,music,movies]
#     list_of_interests=[i for i in list_of_interests if i!=None and i!='']
#     if len(list_of_interests)==0: # случай если профиль не заполнен - ищем интересы по названию групп
#         print('Interests from titles of groups extracting')
#         print_status='Interests from titles of groups extracting'
#         url='https://api.vk.com/method/groups.get?user_id={}&extended=1,fields=uid&v=5.52&access_token={}'.format(user_id,token) # запарсил стену собственной страницы
#         #page= requests.get(url, verify=False, timeout=(1.0, 2)) # allow_redirects (bool)
#         page = requests.get(url, verify=False, timeout=None)
#         #time.sleep(3)
#         data = page.json()
#         list_dict=[]
#         for k,v in data.items():
#             for x,y in v.items():
#                 list_dict.append(y)
#         new_l=list_dict[1]
#         if new_l=='User was deleted or banned':
#             return 0
#         final_list=[]
#         for i in new_l:
#             try:
#                 final_list.append(i.get('name'))
#             except AttributeError:
#                 return 0
#         text_from_groups=[i for i in final_list if i!='']
#         return text_from_groups
#     else:
#         return list_of_interests



def cosine_sim(list_of_interests, interests):
    words_for_vector = []
    for i in list_of_interests:
        text_raw = extract_text(i).replace('\n', ' ')
        regex = re.compile("[А-Яа-я]+")
        text = regex.findall(text_raw.lower())
        text = set(stem_words_array(text))
        words_for_vector.append(text)
    words_for_vector = [x for x in words_for_vector if x]  # подчищенные вектора от пустот
    w2v_vector = []
    for i, l in zip(words_for_vector, list_of_interests):
        if len(i) == 0:
            pass
        else:
            v_sait = vectorize(i, w2v)
            w2v_vector.append([v_sait, l])

    interest_dict = dict()
    for i in interests:
        try:
            cosine = max([sublist[0] for sublist in create_cosine_simularity(i, w2v_vector)])
            interest_dict.update({i: cosine})
        except ValueError:
            pass
    # top_10 = dict(Counter(interest_dict).most_common(10))
    top_10 = dict(Counter(interest_dict).most_common(len(interests_file)))
    if len(top_10) == 0:
        return 0
    else:
        return top_10


def result(message,user_id):
    if ',' in user_id: # вариант когда несколько id перечислено
        bot.reply_to(message, "Извлечение интересов и сравнение со словарями займет некоторое время,подождите")
        user_id = user_id.split(',')
        user_id = [int(i) for i in user_id]
        list_of_interests=[]
        final_result = pd.DataFrame()
        try:
            key_1=[]
            for i in user_id:
                key_1.append(i)
                list_of_interests.append(extraction_interests(int(i)))
            inter_result=list(zip(key_1, list_of_interests))
            key_dict = []
            result = []
            key_id=[]
            for j in tqdm(inter_result):
                for key, value in dict_interests[0].items():
                    interests = value.split(',')
                    interests = [element.strip() for element in interests]  # удаляем пробелы
                    interests = [w.lower() for w in interests]# удаляем пробелы
                    if j[1]==0:
                        key_dict.append(key)
                        key_id.append(j[0])
                        result.append(0) #'Невозможно извлечь'
                    else:
                        key_dict.append(key)
                        key_id.append(j[0])
                        result.append(cosine_sim(j[1], interests))
            result_full = list(zip(key_id,zip(key_dict, result)))
            df = pd.DataFrame(result_full)
            df.columns = ['id', 'dict']
            print(df)
            id_list = df.id.unique()
            for j in tqdm(id_list):
                r = pd.DataFrame(list(df[df['id'] == j].dict))
                r.columns = ['name', 'dict']
                if r.dict[0]==0:
                    res_sen_1 = r.copy()
                    res_sen_1['vk_id'] = j
                    res_sen_1['name'] = False
                    res_sen_1['dict'] = False
                    res_sen_1['mean_res'] = False
                    final_result = final_result.append(res_sen_1)
                else:
                    r['mean_res'] = r['dict'].apply(lambda x: np.array(list(x.values())).mean())
                    res_sen_2 = r.copy()
                    res_sen_2['vk_id'] = j
                    final_result = final_result.append(res_sen_2)
            final_result = final_result[['vk_id', 'name', 'dict', 'mean_res']]
            res_False = final_result[final_result['name'] == False]
            if not res_False.empty:
                res_False = res_False.drop_duplicates(subset='vk_id', keep='first')
                res_True = final_result[final_result['name'] != False]
                list_id = set(res_True.vk_id.tolist())
                if len(list_id)!=0:
                    vk_key = []
                    vk_dict = []
                    for g in list_id:
                        r = res_True[res_True['vk_id'] == g]
                        keys = r.name.tolist()
                        values = r.mean_res.tolist()
                        dictionary = dict(zip(keys, values))
                        dictionary = str([(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get, reverse=True)])
                        vk_key.append(g)
                        vk_dict.append(dictionary)
                    result_full = list(zip(vk_key, vk_dict))
                    result_full = pd.DataFrame(result_full)
                    result_full.columns = ['vk_id', 'result']
                    result_full['result'] = result_full['result'].apply(
                        lambda x: x.replace('(', '').replace('[', '').replace(']', '').replace(')', ''))
                    del res_False['name']
                    del res_False['dict']
                    del res_False['mean_res']
                    res_False['result'] = 'Не удалось извлечь интересы'
                    full_shit = pd.concat([result_full, res_False]).copy()
                else:
                    del res_False['name']
                    del res_False['dict']
                    del res_False['mean_res']
                    res_False['result'] = 'Не удалось извлечь интересы'
                    full_shit=res_False.copy()
            else:
                res_True = final_result[final_result['name'] != False]
                list_id = set(res_True.vk_id.tolist())
                vk_key = []
                vk_dict = []
                for g in list_id:
                    r = res_True[res_True['vk_id'] == g]
                    keys = r.name.tolist()
                    values = r.mean_res.tolist()
                    dictionary = dict(zip(keys, values))
                    dictionary = str([(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get, reverse=True)])
                    vk_key.append(g)
                    vk_dict.append(dictionary)
                result_full = list(zip(vk_key, vk_dict))
                result_full = pd.DataFrame(result_full)
                result_full.columns = ['vk_id', 'result']
                result_full['result'] = result_full['result'].apply(
                    lambda x: x.replace('(', '').replace('[', '').replace(']', '').replace(')', ''))
                full_shit=result_full.copy()
            full_shit.to_excel('result.xlsx', index=False)
            doc = open('result.xlsx', 'rb')
            bot.send_document(message.chat.id, doc)
            os.remove('result.xlsx')
        except Exception as e:
            bot.send_message(message.chat.id, "Произошла ошибка при вводе id {},код ошибки ".format(i) + str(e))
    else:
        try:
            bot.reply_to(message, "Извлечение интересов и сравнение со словарями займет некоторое время,подождите")
            list_of_interests = extraction_interests(int(user_id))
            key_dict = []
            result = []
            for key, value in tqdm(dict_interests[0].items()):
                interests = value.split(',')
                interests = [element.strip() for element in interests]  # удаляем пробелы
                interests = [w.lower() for w in interests]  # удаляем пробелы
                key_dict.append(key)
                result.append(cosine_sim(list_of_interests, interests))
            result_full = list(zip(key_dict, result))
            df_result = pd.DataFrame(result_full)
            df_result.columns = ['name', 'dict']
            df_result['mean_res'] = df_result['dict'].apply(lambda x: np.array(list(x.values())).mean())

            keys = df_result.name.tolist()
            values = df_result.mean_res.tolist()
            dictionary = dict(zip(keys, values))
            dictionary = str(
                [(k, dictionary[k]) for k in sorted(dictionary, key=dictionary.get, reverse=True)]).replace('(',
                                                                                                            '').replace(
                '[', '').replace(']', '').replace(')', '')
            vk_key=[]
            vk_dict = []
            vk_key.append(user_id)
            vk_dict.append(dictionary)
            result_full = list(zip(vk_key, vk_dict))
            result_full = pd.DataFrame(result_full)
            result_full.to_excel('id_{}.xlsx'.format(user_id), index=False)
            # max_value = max(df_result.mean_res)
            # res_sen = df_result[df_result['mean_res'] == max_value]
            # res_sen['vk_id']=user_id
            # res_sen[['vk_id', 'name', 'dict', 'mean_res']].to_excel('id_{}.xlsx'.format(user_id), index=False)
            doc = open('id_{}.xlsx'.format(user_id), 'rb')
            bot.send_document(message.chat.id, doc)
            os.remove('id_{}.xlsx'.format(user_id))
        except Exception as e:
            bot.send_message(message.chat.id, "Произошла ошибка при вводе id {},код ошибки ".format(user_id) + str(e))

def interests(message,user_id):
    bot.reply_to(message, "Вычисление интересов займет некоторое время")
    if ',' in user_id:
        user_id = user_id.split(',')
        user_id = [int(i) for i in user_id]
        appended_data = []
        for i in user_id:
            text_from_groups = extraction_interests(i)
            if text_from_groups == 0 or len(text_from_groups) == 0:
                data = [[i, 'Невозможно извлечь интересы']]
                result = pd.DataFrame(data, columns=['vk_id', 'interests'])
                appended_data.append(result)
            else:
                dict_interests = cosine_sim(text_from_groups, interests_file)
                data = [[i, dict_interests]]
                result = pd.DataFrame(data, columns=['vk_id', 'interests'])
                result['interests'] = result['interests'].astype(str)
                result['interests'] = result['interests'].apply(lambda x: x.replace('{', '').replace('}', ''))
                appended_data.append(result)
        df=pd.concat(appended_data)
        df['interests'] = df['interests'].apply(lambda x: 'Не удалось извлечь интересы' if x == '0' else x)
        df.to_excel('result.xlsx', index=False)
        doc = open('result.xlsx', 'rb')
        bot.send_document(message.chat.id, doc)
        os.remove('result.xlsx')
    else:
        text_from_groups = extraction_interests(user_id)
        if text_from_groups == 0 or len(text_from_groups) == 0:
            bot.reply_to(message, "Не удалось извлечь интересы")
        else:
            dict_interests = cosine_sim(text_from_groups, interests_file)
            data = [[user_id, dict_interests]]
            result = pd.DataFrame(data, columns=['vk_id', 'interests'])
            result['interests'] = result['interests'].astype(str)
            result['interests'] = result['interests'].apply(lambda x: x.replace('{', '').replace('}', ''))
            result.to_excel('result.xlsx', index=False)
            doc = open('result.xlsx', 'rb')
            bot.send_document(message.chat.id, doc)
            os.remove('result.xlsx')


@bot.message_handler(commands=['user_id'])
def start(message):
  sent_msg = bot.send_message(message.chat.id, 'Пожалуйста,введите ID для определения интересов по словарям')
  bot.register_next_step_handler(sent_msg, hello)

@bot.message_handler(commands=['interests'])
def start(message):
  sent_msg = bot.send_message(message.chat.id, 'Пожалуйста,введите ID для определения интересов из списка интересов')
  bot.register_next_step_handler(sent_msg, hello_2)


@bot.message_handler(content_types=["document"])
def handle_file(message):
    try:
        chat_id = message.chat.id
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        src = message.document.file_name;
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        sent_msg = bot.send_message(message.chat.id, 'Сохраняю и обрабатываю файл')
        df=pd.read_excel(src)
        if df.columns[0] == 'id_dict':
            id_list=df.id_dict.tolist()
            id_list=[str(i) for i in id_list]
            user_id=','.join(id_list)
            result(message, user_id)
            os.remove(src)
        elif df.columns[0] == 'id_interests':
            id_list = df.id_interests.tolist()
            id_list = [str(i) for i in id_list]
            user_id = ','.join(id_list)
            interests(message, user_id)
            os.remove(src)
        else:
            bot.reply_to(message, "Пришлите корректный файл либо с колонкой 'id_dict' ,либо 'id_interests'")
    except Exception as e:
        bot.reply_to(message, e)

@bot.message_handler(commands=['help']) # list of commands
def get_help(message):
    bot.send_message(message.chat.id, messages["help-message"])

def hello(message): # для словарей
    user_id=message.text
    result(message, user_id)

def hello_2(message): # для интересов
    user_id=message.text
    interests(message, user_id)
bot.polling()


# TODO убрать парсинг личной страницы-оставить только группы
# TODO извлекать все интересы,а не топ-10
# TODO изменить dict.xlsx - оставить только колонки M и W
# TODO в interests_3.txt вбить слова из M и W


