import time

import microtc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.svm import LinearSVC

import json
from flask import Flask, render_template, jsonify, request, Response, stream_with_context, send_file
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix

app = Flask(__name__)

# Placeholder variables for global values
(f1_g, precision_g, recall_g, accuracy_g, original_classes_g, original_coef_g, original_intercept_g, X_train_g, X_train_tokenized_g,
 X_test_g, X_test_tokenized_g) = None, None, None, None, None, None, None, None, None, None, None
lsvc_target_g = None
text_model_g, text_model_one_token_g = None, None
max_distance, min_distance = None, None
min_weight_g = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/build_header', methods=['POST'])
def build_header():

    # Function to build a header for visualization or formatting

    global f1_g, precision_g, recall_g, accuracy_g, original_classes_g, original_coef_g, original_intercept_g, X_train_g, X_train_tokenized_g, X_test_tokenized_g, X_test_g

    sample_size = request.form.get('sample_size', 0)

    try:
        sample_size = int(sample_size)
    except ValueError:
        result = {
            'processed': False,
            'message': 'Check data types in inputs',
        }
        return jsonify(result)

    json_file_train = request.files.get('jsonFileTrain')
    json_file_test = request.files.get('jsonFileTest')
    if not json_file_train or not json_file_test:
        result = {
            'processed': False,
            'message': 'Not Train/Test file selected!',
        }
        return jsonify(result)

    (X_train, X_test, X_train_shape, X_test_shape, X_train_tokenized, X_test_tokenized, X_train_tokenized_shape,  X_test_tokenized_shape,
     token_list, train_klasses, test_klasses) = build_dataset(sample_size, json_file_train, json_file_test)

    # Initial lienar model
    lsvc = LinearSVC(penalty='l1', C=1.0, dual=False, max_iter=10000, random_state=42)
    lsvc.fit(X_train_tokenized, X_train.klass)
    y_pred = lsvc.predict(X_test_tokenized)
    # initial metrics
    f1 = round(f1_score(X_test.klass, y_pred, average='macro'), 5)
    precision = round(precision_score(X_test.klass, y_pred, average='macro'), 5)
    recall = round(recall_score(X_test.klass, y_pred, average='macro'), 5)
    accuracy = round(accuracy_score(X_test.klass, y_pred), 5)

    f1_g, precision_g, recall_g, accuracy_g = f1, precision, recall, accuracy
    original_classes_g = lsvc.classes_.copy()
    original_coef_g = lsvc.coef_.copy()
    original_intercept_g = lsvc.intercept_.copy()
    X_train_tokenized_g = X_train_tokenized
    X_train_g = X_train
    X_test_tokenized_g = X_test_tokenized
    X_test_g = X_test

    # extras
    sparse_coef = csr_matrix(lsvc.coef_)
    rows, cols = sparse_coef.get_shape()
    total_elements = rows * cols
    density_sparse = f"{(sparse_coef.nnz / total_elements) * 100:.2f}%"
    variance = np.var(sparse_coef.data)

    result = {
        'processed': True,
        'message': 'Successfully processed',
        'sample_size': sample_size,

        'X_train_shape': X_train_shape,
        'X_train_tokenized_shape': X_train_tokenized_shape,
        'X_train_klasses': train_klasses,

        'X_test_shape': X_test_shape,
        'X_test_tokenized_shape': X_test_tokenized_shape,
        'X_test_klasses': test_klasses,

        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,

        'density': f"{density_sparse} : {variance:.2f}%"
    }

    # result in Json
    return jsonify(result)

def build_dataset(sample_size, json_file_train, json_file_test):
    # read the dataset as raw text (train & test) and transform them into tokenized vectors with MicroTc.

    global text_model_g

    # Read the file line by line and decode each line as JSON
    train_data = []
    for line in json_file_train:
        train_data.append(json.loads(line))

    test_data = []
    for line in json_file_test:
        test_data.append(json.loads(line))

    X_train = pd.DataFrame(train_data, columns=['text', 'klass'])
    # if text is list, join the texts
    X_train['text'] = X_train['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    X_test = pd.DataFrame(test_data, columns=['text', 'klass', 'distance', 'pred'])
    X_test['text'] = X_test['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

    X_train_shape = f'({X_train.shape[0]}, {X_train.shape[1]})'
    X_test_shape = f'({X_test.shape[0]}, {X_test.shape[1]})'

    if sample_size <= 0:
        sample_size = 90000

    if len(X_train) > sample_size:
        X_train = X_train.sample(sample_size)
    X_train = X_train.reset_index(drop=True)

    if len(X_test) > sample_size:
        X_test = X_test.sample(sample_size)
    X_test = X_test.reset_index(drop=True)

    train_klasses = ', '.join([f"{key}:{value}" for key, value in X_train.klass.value_counts().items()])
    train_klasses = f'{train_klasses} (Binary)' if len(X_train.klass.unique()) == 2 else f'{train_klasses} (Multiple)'

    test_klasses = ', '.join([f"{key}:{value}" for key, value in X_test.klass.value_counts().items()])
    test_klasses = f'{test_klasses} (Binary)' if len(X_test.klass.unique()) == 2 else f'{test_klasses} (Multiple)'

    # tokenization squema for q-grams
    token_list = [-1, 2, 3, 4] # With q-grams

    # microTc
    text_model = microtc.TextModel(token_list=token_list, del_diac=True, num_option='delete', del_punc=True, url_option='delete', del_dup=False, lc=True, hashtag_option=None, q_grams_words=True)
    text_model = text_model.fit(X_train.text)
    text_model_g = text_model

    # tokenization
    X_train_tokenized =  text_model.transform(X_train.text) # rows are in the same order as the X.index
    X_train_tokenized_shape = f'({X_train_tokenized.shape[0]}, {X_train_tokenized.shape[1]})'

    X_test_tokenized =  text_model.transform(X_test.text)
    X_test_tokenized_shape = f'({X_test_tokenized.shape[0]}, {X_test_tokenized.shape[1]})'

    return X_train, X_test, X_train_shape, X_test_shape, X_train_tokenized, X_test_tokenized, X_train_tokenized_shape,  X_test_tokenized_shape, token_list, train_klasses, test_klasses

@app.route('/threshold_header', methods=['POST'])
def threshold_header():

    global X_test_g, min_decision_value_g, min_weight_g, max_distance, min_distance, lsvc_target_g

    min_decision_value_g = request.form.get('min_decision_value', 0.05)
    min_weight_g = request.form.get('min_weight', 0.005)

    try:
        # Convert the value to float (double in other languages)
        min_decision_value_g = float(min_decision_value_g)
        min_weight_g = float(min_weight_g)
    except ValueError:
        result = {
            'processed': False,
            'message': 'Check data types in inputs',
        }
        return jsonify(result)

    # todo IberLEF2023_PoliticEs_profession sube con 0.1, 0.05...hacer una tabla y grafica
    # todo ojo meoffendes con 0.2 tiene resultados padres

    # Filter coefficients: set to 0 those whose absolute value of the coefficient is less than or equal to threshold_coef
    altered_coef = np.where(np.abs(original_coef_g) < min_weight_g, 0.0, original_coef_g)
    # This filter helps me set to 0 those that do not comply

    # Create a new model with the modified coefficients
    lsvc_target_g = LinearSVC(penalty='l1', C=1.0, dual=False, max_iter=10000, random_state=42)
    lsvc_target_g.classes_ = original_classes_g  # Keep the original classes
    lsvc_target_g.coef_ = altered_coef  # Assign the modified coefficients
    lsvc_target_g.intercept_ = original_intercept_g  # Keep the original intercept

    y_pred_altered = lsvc_target_g.predict(X_test_tokenized_g)
    X_test_g['pred'] = y_pred_altered

    # w⋅x + b
    distances = X_test_tokenized_g @ lsvc_target_g.coef_.T + lsvc_target_g.intercept_

    min_distance, max_distance = np.min(distances), np.max(distances) # use to normalization and html codes
    if distances.shape[1] > 1:
        # If it is a multiclass problem
        distances = np.max(distances, axis=1, keepdims=True)
    X_test_g['distance'] = distances

    # metrics using model with altered coefficients
    f1 = round(f1_score(X_test_g.klass, y_pred_altered, average='macro'), 5)
    precision = round(precision_score(X_test_g.klass, y_pred_altered, average='macro'), 5)
    recall = round(recall_score(X_test_g.klass, y_pred_altered, average='macro'), 5)
    accuracy = round(accuracy_score(X_test_g.klass, y_pred_altered), 5)

    # extras
    sparse_coef = csr_matrix(lsvc_target_g.coef_)
    rows, cols = sparse_coef.get_shape()
    total_elements = rows * cols
    density_sparse = f"{(sparse_coef.nnz / total_elements) * 100:.2f}%"
    variance = np.var(sparse_coef.data)

    if len(lsvc_target_g.coef_) > 1:
        # multiclass
        intercept = ", ".join(f"{k}:{round(i, 8)}" for k, i in zip(lsvc_target_g.classes_, lsvc_target_g.intercept_))
    else:
        intercept = round(lsvc_target_g.intercept_[0], 8)

    result = {
        'processed': True,
        'message': 'Successfully processed',
        'min_decision_value': min_decision_value_g,
        'min_weight': min_weight_g,
        'f1': f1,
        'f1_diff': round((f1 - f1_g) * 100, 2),
        'precision': precision,
        'precision_diff': round((precision - precision_g) * 100, 2),
        'recall': recall,
        'recall_diff': round((recall - recall_g) * 100, 2),
        'accuracy': accuracy,
        'accuracy_diff': round((accuracy - accuracy_g) * 100, 2),
        'intercept_':  intercept,
        'density':  f"{density_sparse} : {variance:.2f}%"
    }

    return jsonify(result)

# Normalize values ​​between 0 and 1 (considering negative and positive values)
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Map a normalized value to an HTML color
def value_to_color(value, min_value, max_value):
    if value > 0:
        # Normalize for positive values ​​and map from low red (#FFCCCC) to strong red (#FF0000)
        normalized_value = normalize(value, 0, max_value)
        r = 255  # Rojo constante
        g = int(204 * (1 - normalized_value))  # Verde disminuye para intensificar el rojo
        b = int(204 * (1 - normalized_value))  # Azul disminuye para intensificar el rojo
    elif value < 0:
        # Normalize for negative values ​​and map from strong blue (#0000FF) to low blue (#CCCCFF)
        normalized_value = normalize(value, min_value, 0)
        r = int(204 * normalized_value)  # Rojo aumenta para aclarar el azul
        g = int(204 * normalized_value)  # Verde aumenta para aclarar el azul
        b = 255  # Azul constante
    else:
        # Color blanco para el valor 0
        r, g, b = 255, 255, 255  # Blanco

    return f'#{r:02X}{g:02X}{b:02X}'

def tracing_grams(tweet_idx):
    # For each word in the tweet, apply microTc with the same tokenization scheme to map the q-grams to their corresponding source words.

    X_instance = X_test_g.loc[tweet_idx, ['text', 'klass', 'distance', 'pred']].values
    tweet_txt, true_klass, distance, pred_klass = X_instance[0], X_instance[1], X_instance[2], X_instance[3]

    tweet_tokens_one_words = text_model_one_token_g.tokenize(tweet_txt)

    tf_instance = X_test_tokenized_g[tweet_idx].toarray().flatten()  # VECTOR TFIDF instance

    if len(lsvc_target_g.coef_) == 1:
        weights_full = tf_instance * lsvc_target_g.coef_[0]
    else: # multiple
        index_pred_klass = list(lsvc_target_g.classes_).index(pred_klass) if isinstance(pred_klass, str) else pred_klass
        weights_full = tf_instance * lsvc_target_g.coef_[index_pred_klass]  # WINNING VECTOR only that of the selected KLASSE

    indices_no_cero = np.nonzero(weights_full)[0]

    tokens = [text_model_g.id2token[i] for i in indices_no_cero]
    weights = weights_full[indices_no_cero]
    # assemble tokens
    vector_tokens = dict(zip(tokens, weights))

    # tweet_words = tweet_txt.split() # Deprecated
    tokens_weights = {}
    min_weight, max_weight = 0.00, 0.00

    solo_palabras = {}

    for word in tweet_tokens_one_words:
        arreglo_de_pesos_per_word = np.array([])
        for token in text_model_g.tokenize(word):
            try:
                weight = vector_tokens[token]
                tokens_weights[token] = weight
                arreglo_de_pesos_per_word = np.append(arreglo_de_pesos_per_word, weight)
                if weight < min_weight:
                    min_weight = weight
                elif weight > max_weight:
                    max_weight = weight
            except Exception as e:
                # add the words even if they are 0
                if not token.startswith('q:'):
                    tokens_weights[token] = 0
                    #continue
                #continue
        weight_per_word = np.sum(arreglo_de_pesos_per_word) if len(arreglo_de_pesos_per_word) > 0 else 0
        if weight_per_word < min_weight:
            min_weight = weight_per_word
        elif weight_per_word > max_weight:
            max_weight = weight_per_word
        solo_palabras[word] = weight_per_word

    print(tweet_idx, distance, min_weight, max_weight)

    # deprecated
    #words_weights = {key: [value,  value_to_color(value, min_coef, max_coef)] for key, value in tokens_weights.items() if not key.startswith('q:')}

    words_weights = {key: [value,  value_to_color(value, min_weight, max_weight)] for key, value in solo_palabras.items()}

    decision_color_html = value_to_color(distance, min_distance, max_distance)

    return words_weights, tokens_weights, decision_color_html

def htmlizer():
    # visualize structure

    global text_model_one_token_g
    text_model_one_token_g = microtc.TextModel(token_list=[-1], del_punc=True, del_diac=True, lc=True, usr_option='group')

    tr_list = []  # Initialize as list

    for idx, row in X_test_g.iterrows():

        #if idx < 1020:
        #    continue

        distance = row['distance']

        if abs(distance) >= min_decision_value_g:

            tweet = row['text']
            true_klass = row['klass']
            pred_klass = row['pred']

            # words_weights has the accumulated weights per word and it is what is displayed in the tweet
            # tokens_weights has all the unit weights per token (q_grams and words) and is the tooltip
            words_weights, tokens_weights, decision_color_html = tracing_grams(idx)

            tr_row = ''

            unique_words_weights = {}
            seen_combinations = set()

            for word, (weight, color) in words_weights.items():
                if (word, weight) not in seen_combinations:
                    seen_combinations.add((word, weight))
                    unique_words_weights[word] = (weight, color)

            for word, (weight, color) in unique_words_weights.items():
                if abs(weight) >= min_weight_g:

                    tkn_frag_aggs = {}
                    for tkn_frag in text_model_g.tokenize(word):
                        weight_frag = tokens_weights.get(tkn_frag, None)
                        if weight_frag:
                            tkn_frag_aggs[tkn_frag] = weight_frag

                    frag_tip = """<table class='tooltip_text'> """
                    frag_tip += f"<tr><td><strong>{word}</strong></td><td><strong/>∑{round(weight, 5):+.5f}</strong></td></tr>"
                    for g, v in tkn_frag_aggs.items():
                        # grams = {key.replace('q:', ''): value for key, value in grams.items()}
                        if g.startswith('q:'):
                            frag_tip += f"<tr><td>{g.replace('q:', '')}</td><td>{round(v, 5):+.5f}</td></tr>"
                    frag_tip += "</table>"

                    span_word = (f"<span class='sp_word tooltip' style='background-color:{color}; '>{word} "
                                 f"{frag_tip}</span>")

                    # word with +/- weight
                    #span_word = (f"<span class='sp_word tooltip' style='background-color:{color}; '>{word} "
                    #             f"<span class='tooltip_text' style='font-size: smaller; width: unset;'>{round(weight, 6)} </span></span>")



                else:
                    # word zero
                    span_word = f"<span class='sp_word' style='color: unset;'>{word}</span>"
                tr_row += span_word

            weight_total = sum(tokens_weights.values())
            #tokens_weights = {key : value for key, value in tokens_weights.items() if abs(value) >= min_weight_g}
            weight_umbral = sum(tokens_weights.values())
            tokens_weights = dict(sorted(tokens_weights.items(), key=lambda item: abs(item[1]), reverse=True))
            num_de_qgrams = 15
            if len(tokens_weights) > num_de_qgrams:
                primeros_grams = dict(list(tokens_weights.items())[:num_de_qgrams])
                restantes = list(tokens_weights.items())[num_de_qgrams:]
                peso_restante = sum([v for k, v in restantes])
                primeros_grams[f'{len(restantes)} more...'] = peso_restante
                tokens_weights = primeros_grams

            span_tip = """<table class='tooltip_text'> """
            span_tip += f"<tr><td><strong>Weight</strong></td><td><strong/>{round(weight_total, 5):+.5f}</strong></td></tr>"
            span_tip += f"<tr><td><strong>Threshold</strong></td><td><strong/>{round(weight_umbral, 5):+.5f}</strong></td></tr>"
            for g, v in tokens_weights.items():
                # grams = {key.replace('q:', ''): value for key, value in grams.items()}
                if g.startswith('q:'):
                    span_tip += f"<tr><td>{g.replace('q:', '')}</td><td>{round(v, 4):+.4f}</td></tr>"
                else:
                    span_tip += f"<tr><td style='font-weight: 600;'><em>{g}</em></td><td>{round(v, 4):+.4f}</td></tr>"
            span_tip += "</table>"

            #pred_legend = f"<span class='tooltip' >{pred_klass} {round(decision_value_pred, 5):.5f} {span_tip}</span>"

            if pred_klass == true_klass:
                true_legend = f"<span class='span_mark'>&nbsp</span>" #  ✔
            else:
                true_legend = f"<span class='span_mark'>✘</span>"

            pred_td = f"<span class='tooltip'>" \
                      f"   <span class='span_pred_value' style='background-color: {decision_color_html};' >{round(distance, 5):+.5f}</span> " \
                      f"   <span class='span_pred_klass' >{pred_klass}</span>" \
                      f"   <span class='span_true_klass' >{true_klass}</span>" \
                      f"{true_legend}{span_tip}</span>"

            tr_row = f"<tr class='tr_row'><td class='td_index'>{idx}</td><td class='td_words'>{tr_row}</td><td class='td_pred'>{pred_td}</td></tr>"

            tr_list.append(tr_row)

            #table_rows.append(f"<tr><td>{idx}</td><td>{tweet}</td><td>{true_klass}</td><td>{distance}</td><td>{pred_klass}</td></tr>")

    return tr_list

def generate_chunks():

    table_rows = htmlizer()

    total_row = len(table_rows)
    chunk_size = 400
    yield f"data: __START__\n\n"
    for i in range(0, total_row, chunk_size):
        chunk = table_rows[i:i + chunk_size]
        yield f"data: {''.join(chunk)}\n\n"
        time.sleep(1)
    yield "data: __COMPLETE__\n\n"  # Signal to indicate that the shipment was completed

@app.route('/stream-data-html')
def stream():

    return Response(stream_with_context(generate_chunks()), content_type='text/event-stream')

@app.route('/download-json')
def download_json():

    def json_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return str(obj)  # Convert numeric types to string
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    tokens_data = []

    for idx, row in X_test_g.iterrows():

        distance = row['distance']

        if abs(distance) >= min_decision_value_g:

            tweet = row['text']
            true_klass = row['klass']
            pred_klass = row['pred']

            words_weights, tokens_weights, decision_color_html = tracing_grams(idx)

            tokens_data.append({str(idx): {'prediction_klass': str(pred_klass), 'decision_value': str(distance), 'true_klass': true_klass, 'data': tokens_weights}})

    filename = 'data.json'

    # Save the list of dictionaries to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(tokens_data, json_file, default=json_serializer, indent=4)

    # Submit the file for download
    return send_file(filename, as_attachment=True)



def view_json():

    X_filtered = X[X.text.str.len() < 280].copy()
    tokens_data = []

    hard_negatives = []

    def json_serializer(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return str(obj)  # Convert numeric types to string
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    for index, row in X_filtered.iterrows():

        pred_klass, decision_value_pred, true_klass, words_weights, tokens_weights, decision_color_html = tracing_grams(index)
        tokens_data.append({str(index): {'prediction_klass': str(pred_klass), 'decision_value': str(decision_value_pred),
                            'true_klass': true_klass, 'data': tokens_weights}})

        rec = {'idx': str(index), 'txt': row['text'], 'true': str(true_klass), 'pred': str(pred_klass), 'decision': str(decision_value_pred) }
        hard_negatives.append({'idx': str(index), 'txt': row['text'], 'true': str(true_klass), 'pred': str(pred_klass), 'decision': str(decision_value_pred) })

    filename = 'data.json'

    # Save the list of dictionaries to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(tokens_data, json_file, default=json_serializer, indent=4)

    with open('meoffendes_hard_negatives.json', 'w') as json_file:
        json.dump(hard_negatives, json_file, default=json_serializer, indent=4)


    # Send the file as a reply to view it in a new tab
    #return send_file(filename, mimetype='application/json')

    # Submit the file for download
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
