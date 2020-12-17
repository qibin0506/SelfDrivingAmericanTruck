from pynput import keyboard
import threading

key_callback = None

lock = threading.Lock()

key_map = {
    'w': [1, 0, 0, 0, 0, 0, 0, 0, 0],
    'a': [0, 1, 0, 0, 0, 0, 0, 0, 0],
    's': [0, 0, 1, 0, 0, 0, 0, 0, 0],
    'd': [0, 0, 0, 1, 0, 0, 0, 0, 0],
    'wa': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'aw': [0, 0, 0, 0, 1, 0, 0, 0, 0],
    'wd': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'dw': [0, 0, 0, 0, 0, 1, 0, 0, 0],
    'sa': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'as': [0, 0, 0, 0, 0, 0, 1, 0, 0],
    'sd': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'ds': [0, 0, 0, 0, 0, 0, 0, 1, 0],
    'no': [0, 0, 0, 0, 0, 0, 0, 0, 1],
}

cur_key = None


def get_cur_key():
    if cur_key is None or cur_key is '' or cur_key not in key_map:
        return 'no'

    return cur_key


def get_encoded_key(key):
    if key is None or key is '' or key not in key_map:
        return [0, 0, 0, 0, 0, 0, 0, 0, 1]

    return key_map[key]


def on_press(key):
    if key == keyboard.Key.esc:
        if key_callback is not None:
            key_callback(key)

        return False

    if key == keyboard.Key.tab:
        if key_callback is not None:
            key_callback(key)

        return True

    lock.acquire()
    global cur_key

    try:
        if cur_key is None:
            cur_key = key.char
        else:
            if key.char not in cur_key:
                cur_key += key.char
    except:
        pass

    # print('current: {}'.format(cur_key))
    lock.release()


def on_release(key):
    if key == keyboard.Key.esc:
        return False

    if key == keyboard.Key.tab:
        return True

    lock.acquire()
    global cur_key

    try:
        if cur_key is key.char:
            cur_key = None
        else:
            cur_key = str.replace(cur_key, key.char, '')
    except:
        pass

    # print('current: {}'.format(cur_key))
    lock.release()


def listen(keycallback):
    global key_callback
    key_callback = keycallback

    def start():
        with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
            listener.join()

    threading.Thread(target=start).start()
