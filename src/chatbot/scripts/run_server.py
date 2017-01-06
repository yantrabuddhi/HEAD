#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import os
import logging
import datetime as dt
import json
import shutil

import sys
CWD = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(CWD, '../src'))

if 'HR_CHARACTER_PATH' not in os.environ:
    os.environ['HR_CHARACTER_PATH'] = os.path.join(CWD, 'characters')

from chatbot.server.config import CHATBOT_LOG_DIR, HISTORY_DIR

if not os.path.isdir(CHATBOT_LOG_DIR):
    os.makedirs(CHATBOT_LOG_DIR)
LOG_CONFIG_FILE = '{}/chatbot_server_{}.log'.format(CHATBOT_LOG_DIR,
                                                    dt.datetime.strftime(dt.datetime.now(), '%Y%m%d%H%M%S'))
link_log_fname = os.path.join(CHATBOT_LOG_DIR, 'chatbot_server_latest.log')
if os.path.islink(link_log_fname):
    os.unlink(link_log_fname)
os.symlink(LOG_CONFIG_FILE, link_log_fname)
fh = logging.FileHandler(LOG_CONFIG_FILE)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    '[%(name)s][%(levelname)s] %(asctime)s: %(message)s')
fh.setFormatter(formatter)
sh.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(fh)
root_logger.addHandler(sh)

from chatbot.server.auth import requires_auth
from chatbot.server.auth import check_auth, authenticate

from flask import Flask, request, Response, send_from_directory

from chatbot.server.chatbot_agent import (
    ask, list_character, session_manager, set_weights, set_context,
    dump_history, dump_session, add_character, list_character_names,
    rate_answer, get_context, said, remove_context)
from chatbot.stats import history_stats

json_encode = json.JSONEncoder().encode
app = Flask(__name__)
VERSION = 'v1.1'
ROOT = '/{}'.format(VERSION)
INCOMING_DIR = os.path.expanduser('~/.hr/aiml/incoming')

logger = logging.getLogger('hr.chatbot.server')
app.config['UPLOAD_FOLDER'] = os.path.expanduser('~/.hr/aiml')


@app.route(ROOT + '/<path:path>')
def send_client(path):
    return send_from_directory('public', path)


@app.route(ROOT + '/client', methods=['GET'])
def client():
    return send_from_directory('public', 'client.html')


@app.route(ROOT + '/chat', methods=['GET'])
@requires_auth
def _chat():
    data = request.args
    question = data.get('question')
    session = data.get('session')
    lang = data.get('lang', 'en')
    query = data.get('query', 'false')
    query = query.lower() == 'true'
    response, ret = ask(question, lang, session, query)
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")


@app.route(ROOT + '/batch_chat', methods=['POST'])
def _batch_chat():
    auth = request.form.get('Auth')
    if not auth or not check_auth(auth):
        return authenticate()

    questions = request.form.get('questions')
    questions = json.loads(questions)
    session = request.form.get('session')
    lang = request.form.get('lang', 'en')
    responses = []
    for idx, question in questions:
        response, ret = ask(str(question), lang, session)
        responses.append((idx, response, ret))
    return Response(json_encode({'ret': 0, 'response': responses}),
                    mimetype="application/json")

@app.route(ROOT + '/said', methods=['GET'])
@requires_auth
def _said():
    data = request.args
    session = data.get('session')
    message = data.get('message')
    ret, response = said(session, message)
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")

@app.route(ROOT + '/rate', methods=['GET'])
@requires_auth
def _rate():
    data = request.args
    response = ''
    try:
        ret = rate_answer(data.get('session'), int(
            data.get('index')), data.get('rate'))
    except Exception as ex:
        response = ex.message
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")


@app.route(ROOT + '/chatbots', methods=['GET'])
@requires_auth
def _chatbots():
    data = request.args
    lang = data.get('lang', None)
    session = data.get('session')
    characters = list_character(lang, session)
    return Response(json_encode({'ret': 0, 'response': characters}),
                    mimetype="application/json")


@app.route(ROOT + '/bot_names', methods=['GET'])
@requires_auth
def _bot_names():
    names = list_character_names()
    return Response(json_encode({'ret': 0, 'response': names}),
                    mimetype="application/json")


@app.route(ROOT + '/start_session', methods=['GET'])
@requires_auth
def _start_session():
    botname = request.args.get('botname')
    user = request.args.get('user')
    test = request.args.get('test', 'false')
    refresh = request.args.get('refresh', 'false')
    test = test.lower() == 'true'
    refresh = refresh.lower() == 'true'
    sid = session_manager.start_session(
        user=user, key=botname, test=test, refresh=refresh)
    sess = session_manager.get_session(sid)
    sess.sdata.botname = botname
    sess.sdata.user = user
    return Response(json_encode({'ret': 0, 'sid': str(sid)}),
                    mimetype="application/json")


@app.route(ROOT + '/sessions', methods=['GET'])
@requires_auth
def _sessions():
    sessions = session_manager.list_sessions()
    return Response(json_encode({'ret': 0, 'response': sessions}),
                    mimetype="application/json")

@app.route(ROOT + '/set_weights', methods=['GET'])
@requires_auth
def _set_weights():
    data = request.args
    lang = data.get('lang', None)
    param = data.get('param')
    sid = data.get('session')
    ret, response = set_weights(param, lang, sid)
    if ret:
        sess = session_manager.get_session(sid)
        if sess and hasattr(sess.sdata, 'weights'):
            logger.info("Set weights {} successfully".format(sess.sdata.weights))
    else:
        logger.info("Set weights failed.")
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")

@app.route(ROOT + '/set_context', methods=['GET'])
@requires_auth
def _set_context():
    data = request.args
    context_str = data.get('context')
    context = {}
    for tok in context_str.split(','):
        k, v = tok.split('=')
        context[k.strip()] = v.strip()
    sid = data.get('session')
    ret, response = set_context(context, sid)
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")


@app.route(ROOT + '/remove_context', methods=['GET'])
@requires_auth
def _remove_context():
    data = request.args
    keys = data.get('keys')
    keys = keys.split(',')
    sid = data.get('session')
    ret, response = remove_context(keys, sid)
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")

@app.route(ROOT + '/get_context', methods=['GET'])
@requires_auth
def _get_context():
    data = request.args
    sid = data.get('session')
    ret, response = get_context(sid)
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")

@app.route(ROOT + '/upload_character', methods=['POST'])
def _upload_character():
    auth = request.form.get('Auth')
    if not auth or not check_auth(auth):
        return authenticate()
    try:
        user = request.form.get('user')
        zipfile = request.files.get('zipfile')
        lang = request.files.get('lang')

        saved_dir = os.path.join(INCOMING_DIR, user)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        # Clean the incoming directory
        for f in os.listdir(saved_dir):
            f = os.path.join(saved_dir, f)
            if os.path.isfile(f):
                os.unlink(f)
            else:
                shutil.rmtree(f, True)

        saved_zipfile = os.path.join(saved_dir, zipfile.filename)
        zipfile.save(saved_zipfile)
        logger.info("Get zip file {}".format(zipfile.filename))

        from loader import AIMLCharacterZipLoader
        characters = AIMLCharacterZipLoader.load(
            saved_zipfile, saved_dir, 'upload')
        ret, response = True, "Done"
        for character in characters:
            character.local = False
            character.id = '{}/{}'.format(user, character.id)
            _ret, _response = add_character(character)
            if not _ret:
                ret = _ret
                response = _response
        os.remove(saved_zipfile)

        return Response(json_encode({
            'ret': ret,
            'response': response
        }),
            mimetype="application/json")

    except Exception as ex:
        return Response(json_encode({
            'ret': False,
            'response': str(ex)
        }),
            mimetype="application/json")


@app.route('/log')
def _log():
    def generate():
        with open(LOG_CONFIG_FILE) as f:
            for row in f:
                yield row
    return Response(generate(), mimetype='text/plain')


@app.route(ROOT + '/reset_session', methods=['GET'])
@requires_auth
def _reset_session():
    data = request.args
    sid = data.get('session')
    if session_manager.has_session(sid):
        session_manager.reset_session(sid)
        ret, response = True, "Session reset"
    else:
        ret, response = False, "No such session"
    return Response(json_encode({
        'ret': ret,
        'response': response
    }),
        mimetype="application/json")


@app.route(ROOT + '/dump_history', methods=['GET'])
def _dump_history():
    try:
        dump_history()
        ret, response = True, "Success"
    except Exception:
        ret, response = False, "Failure"
    return Response(json_encode({
        'ret': ret,
        'response': response
    }),
        mimetype="application/json")


@app.route(ROOT + '/dump_session', methods=['GET'])
@requires_auth
def _dump_session():
    try:
        data = request.args
        sid = data.get('session')
        fname = dump_session(sid)
        session_manager.remove_session(sid)
        if fname:
            return send_from_directory(os.path.dirname(fname), os.path.basename(fname))
        else:
            return '', 404
    except Exception as ex:
        logger.error("Dump error {}".format(ex))
        return '', 500


@app.route(ROOT + '/chat_history', methods=['GET'])
@requires_auth
def _chat_history():
    history_stats(HISTORY_DIR, 7)
    history_file = os.path.join(HISTORY_DIR, 'last_7_days.csv')
    if os.path.isfile(history_file):
        return send_from_directory(HISTORY_DIR, os.path.basename(history_file))
    else:
        return '', 404


@app.route(ROOT + '/ping', methods=['GET'])
def _ping():
    return Response(json_encode({'ret': 0, 'response': 'pong'}),
                    mimetype="application/json")


@app.route(ROOT + '/stats', methods=['GET'])
@requires_auth
def _stats():
    try:
        data = request.args
        days = int(data.get('lookback', 7))
        dump_history()
        response = history_stats(HISTORY_DIR, days)
        ret = True
    except Exception as ex:
        ret, response = False, {'err_msg': str(ex)}
        logger.error(ex)
    return Response(json_encode({'ret': ret, 'response': response}),
                    mimetype="application/json")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8001
    if 'HR_CHATBOT_SERVER_EXT_PATH' in os.environ:
        sys.path.insert(0, os.path.expanduser(
            os.environ['HR_CHATBOT_SERVER_EXT_PATH']))
        import ext
        ext.load(app, ROOT)
    app.run(host='0.0.0.0', debug=False, use_reloader=False, port=port)
