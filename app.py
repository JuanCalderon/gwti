# This Flask application provides a web interface for text classification analysis.
# It allows users to upload training and testing datasets, train a LinearSVC model,
# visualize token weights, and analyze model predictions with adjustable thresholds.
# Key features include dynamic filtering of model coefficients, visualization of
# token contributions to predictions, and data download capabilities.

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

# Global dictionary to store application data.
# This dictionary acts as a central store for shared data across different
# parts of the application, such as model parameters, datasets, and UI settings.
app_data = {}

# Placeholder variables for global values - REMOVED as they are now in app_data
# (f1_g, precision_g, recall_g, accuracy_g, original_classes_g, original_coef_g, original_intercept_g, X_train_g, X_train_tokenized_g,
#  X_test_g, X_test_tokenized_g) = None, None, None, None, None, None, None, None, None, None, None
# lsvc_target_g = None
# text_model_g, text_model_one_token_g = None, None
# max_distance, min_distance = None, None
# min_weight_g = None

@app.route('/')
def index():
    """Serves the main HTML page of the application."""
    return render_template('index.html')


@app.route('/build_header', methods=['POST'])
def build_header():
    """
    Processes uploaded train/test JSON files, builds datasets, trains an initial
    LinearSVC model, and calculates initial performance metrics.
    Stores the datasets, model parameters, and metrics in `app_data` for later use.

    Request form data:
        sample_size (int, optional): Maximum number of samples to use from train/test sets.
        jsonFileTrain (FileStorage): Training data in JSONL format.
        jsonFileTest (FileStorage): Test data in JSONL format.

    Returns:
        JSON response with processing status, dataset shapes, class distributions,
        and initial model performance metrics (F1, precision, recall, accuracy).
    """
    global app_data

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

    # Store original model parameters, metrics, and datasets in app_data
    app_data['f1_original'] = f1
    app_data['precision_original'] = precision
    app_data['recall_original'] = recall
    app_data['accuracy_original'] = accuracy
    app_data['original_classes'] = lsvc.classes_.copy()
    app_data['original_coef'] = lsvc.coef_.copy()
    app_data['original_intercept'] = lsvc.intercept_.copy()
    app_data['X_train_tokenized'] = X_train_tokenized # TF-IDF vectors for training text
    app_data['X_train'] = X_train # DataFrame for training data
    app_data['X_test_tokenized'] = X_test_tokenized # TF-IDF vectors for test text
    app_data['X_test'] = X_test # DataFrame for test data

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
    """
    Reads train and test JSONL files, processes the text data, and tokenizes it using MicroTC.
    Stores the main MicroTC text model in `app_data`.

    Args:
        sample_size (int): Maximum number of samples to use. If 0, a default is used.
        json_file_train (FileStorage): Uploaded training data file.
        json_file_test (FileStorage): Uploaded test data file.

    Returns:
        tuple: Contains processed train/test DataFrames, their shapes, tokenized versions,
               token_list used, and class distribution strings.
    """
    global app_data

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

    # Tokenization schema for q-grams:
    # -1: individual words (unigrams)
    # 2: bigrams of characters
    # 3: trigrams of characters
    # 4: 4-grams of characters
    token_list = [-1, 2, 3, 4] # With q-grams

    # Initialize and fit MicroTC text model.
    # Parameters include:
    #   del_diac: remove diacritics
    #   num_option: 'delete' numbers
    #   del_punc: delete punctuation
    #   url_option: 'delete' URLs
    #   del_dup: do not delete duplicate characters in sequence
    #   lc: lowercase text
    #   hashtag_option: None (keep hashtags as is)
    #   q_grams_words: True (generate q-grams from words, not just characters)
    text_model = microtc.TextModel(token_list=token_list, del_diac=True, num_option='delete', del_punc=True, url_option='delete', del_dup=False, lc=True, hashtag_option=None, q_grams_words=True)
    text_model = text_model.fit(X_train.text)
    app_data['text_model'] = text_model # Store the primary text model

    # tokenization
    X_train_tokenized =  app_data['text_model'].transform(X_train.text) # rows are in the same order as the X.index
    X_train_tokenized_shape = f'({X_train_tokenized.shape[0]}, {X_train_tokenized.shape[1]})'

    X_test_tokenized =  app_data['text_model'].transform(X_test.text)
    X_test_tokenized_shape = f'({X_test_tokenized.shape[0]}, {X_test_tokenized.shape[1]})'

    return X_train, X_test, X_train_shape, X_test_shape, X_train_tokenized, X_test_tokenized, X_train_tokenized_shape,  X_test_tokenized_shape, token_list, train_klasses, test_klasses

@app.route('/threshold_header', methods=['POST'])
def threshold_header():
    """
    Applies new thresholds (minimum decision value, minimum coefficient weight)
    to the model stored in `app_data`. It filters model coefficients, re-evaluates
    the model on the test set, and calculates new performance metrics and feature distances.

    Request form data:
        min_decision_value (float, optional): Minimum absolute decision value for a prediction to be considered significant.
        min_weight (float, optional): Minimum absolute coefficient weight for a feature to be kept.

    Returns:
        JSON response with processing status, applied thresholds, new performance metrics
        (and their difference from original metrics), updated intercept, and model density.
    """
    global app_data

    # Store UI-provided thresholds in app_data
    app_data['min_decision_value'] = request.form.get('min_decision_value', 0.05)
    app_data['min_weight'] = request.form.get('min_weight', 0.005)

    try:
        # Convert the value to float (double in other languages)
        app_data['min_decision_value'] = float(app_data['min_decision_value'])
        app_data['min_weight'] = float(app_data['min_weight'])
    except ValueError:
        result = {
            'processed': False,
            'message': 'Check data types in inputs',
        }
        return jsonify(result)

    # todo IberLEF2023_PoliticEs_profession sube con 0.1, 0.05...hacer una tabla y grafica
    # todo ojo meoffendes con 0.2 tiene resultados padres

    # Filter model coefficients: set coefficients to 0 if their absolute value is less than app_data['min_weight'].
    # This creates a sparser model by removing less important features.
    altered_coef = np.where(np.abs(app_data['original_coef']) < app_data['min_weight'], 0.0, app_data['original_coef'])
    # This filter helps me set to 0 those that do not comply

    # Create a new LinearSVC model ('lsvc_target') using the altered coefficients.
    # This model will be used for predictions and analysis based on the new thresholds.
    lsvc_target = LinearSVC(penalty='l1', C=1.0, dual=False, max_iter=10000, random_state=42)
    lsvc_target.classes_ = app_data['original_classes']  # Keep the original classes
    lsvc_target.coef_ = altered_coef  # Assign the modified (filtered) coefficients
    lsvc_target.intercept_ = app_data['original_intercept']  # Keep the original intercept
    app_data['lsvc_target'] = lsvc_target # Store the new, thresholded model

    y_pred_altered = app_data['lsvc_target'].predict(app_data['X_test_tokenized'])
    app_data['X_test']['pred'] = y_pred_altered # Update predictions in the test DataFrame

    # Calculate decision function values (distances to the hyperplane) for each instance.
    # Formula: w⋅x + b, where w is coefficient vector, x is feature vector, b is intercept.
    distances = app_data['X_test_tokenized'] @ app_data['lsvc_target'].coef_.T + app_data['lsvc_target'].intercept_

    # Store min/max distances for visualization scaling (e.g., color mapping)
    app_data['min_distance_viz'] = np.min(distances)
    app_data['max_distance_viz'] = np.max(distances)
    if distances.shape[1] > 1:
        # For multiclass, use the max distance among all classes for each instance
        distances = np.max(distances, axis=1, keepdims=True)
    app_data['X_test']['distance'] = distances # Store calculated distances in the test DataFrame

    # metrics using model with altered coefficients
    f1 = round(f1_score(app_data['X_test'].klass, y_pred_altered, average='macro'), 5)
    precision = round(precision_score(app_data['X_test'].klass, y_pred_altered, average='macro'), 5)
    recall = round(recall_score(app_data['X_test'].klass, y_pred_altered, average='macro'), 5)
    accuracy = round(accuracy_score(app_data['X_test'].klass, y_pred_altered), 5)

    # extras
    sparse_coef = csr_matrix(app_data['lsvc_target'].coef_)
    rows, cols = sparse_coef.get_shape()
    total_elements = rows * cols
    density_sparse = f"{(sparse_coef.nnz / total_elements) * 100:.2f}%"
    variance = np.var(sparse_coef.data)

    if len(app_data['lsvc_target'].coef_) > 1:
        # multiclass
        intercept = ", ".join(f"{k}:{round(i, 8)}" for k, i in zip(app_data['lsvc_target'].classes_, app_data['lsvc_target'].intercept_))
    else:
        intercept = round(app_data['lsvc_target'].intercept_[0], 8)

    result = {
        'processed': True,
        'message': 'Successfully processed',
        'min_decision_value': app_data['min_decision_value'],
        'min_weight': app_data['min_weight'],
        'f1': f1,
        'f1_diff': round((f1 - app_data['f1_original']) * 100, 2),
        'precision': precision,
        'precision_diff': round((precision - app_data['precision_original']) * 100, 2),
        'recall': recall,
        'recall_diff': round((recall - app_data['recall_original']) * 100, 2),
        'accuracy': accuracy,
        'accuracy_diff': round((accuracy - app_data['accuracy_original']) * 100, 2),
        'intercept_':  intercept,
        'density':  f"{density_sparse} : {variance:.2f}%"
    }

    return jsonify(result)

# Normalize values ​​between 0 and 1 (considering negative and positive values)
def normalize(value, min_value, max_value):
    """Normalizes a value to a 0-1 range given min and max bounds."""
    if max_value == min_value: # Avoid division by zero if all values are the same
        return 0.5 # Or any other appropriate default for this case
    return (value - min_value) / (max_value - min_value)

# Map a normalized value to an HTML color
def value_to_color(value, min_value, max_value):
    """
    Maps a numerical value to an HTML hex color code.
    Positive values are mapped to shades of red (higher value = stronger red).
    Negative values are mapped to shades of blue (lower value = stronger blue).
    Zero is mapped to white.
    Uses `normalize` for scaling.

    Args:
        value (float): The value to map to a color.
        min_value (float): The minimum bound for normalization (for negative values).
        max_value (float): The maximum bound for normalization (for positive values).

    Returns:
        str: HTML hex color code (e.g., '#FF0000').
    """
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
    """
    Calculates and traces the weights of q-grams and words for a specific tweet instance.
    It determines how much each token (word or q-gram) contributes to the prediction score.

    Args:
        tweet_idx (int): The index of the tweet in the `app_data['X_test']` DataFrame.

    Returns:
        tuple:
            - words_weights (dict): Maps each word in the tweet to its aggregated weight and color.
                                    Structure: {word: [aggregated_weight, color_code]}
            - tokens_weights (dict): Maps each individual token (word or q-gram) to its weight.
                                     Structure: {token_string: weight}
            - decision_color_html (str): HTML color code representing the tweet's overall decision value.
    """
    # Retrieve instance data from app_data['X_test']
    X_instance = app_data['X_test'].loc[tweet_idx, ['text', 'klass', 'distance', 'pred']].values
    tweet_txt, true_klass, distance, pred_klass = X_instance[0], X_instance[1], X_instance[2], X_instance[3]

    # Tokenize the tweet into individual words using a simpler model (app_data['text_model_one_token'])
    # This is used to iterate over "words" in the tweet for attributing q-gram weights.
    tweet_tokens_one_words = app_data['text_model_one_token'].tokenize(tweet_txt)

    # Get the TF-IDF vector for the specific tweet instance
    tf_instance = app_data['X_test_tokenized'][tweet_idx].toarray().flatten()

    # Calculate full weights: element-wise product of TF-IDF values and model coefficients.
    # This shows the contribution of each feature (token in the vocabulary) to the decision score.
    if len(app_data['lsvc_target'].coef_) == 1: # Binary classification
        weights_full = tf_instance * app_data['lsvc_target'].coef_[0]
    else: # Multiclass classification
        # Get coefficients for the predicted class
        index_pred_klass = list(app_data['lsvc_target'].classes_).index(pred_klass) if isinstance(pred_klass, str) else pred_klass
        weights_full = tf_instance * app_data['lsvc_target'].coef_[index_pred_klass]

    # Identify non-zero weights and their corresponding tokens from the main text_model vocabulary
    indices_no_cero = np.nonzero(weights_full)[0]
    tokens = [app_data['text_model'].id2token[i] for i in indices_no_cero] # Map token IDs to token strings
    weights = weights_full[indices_no_cero]
    # `vector_tokens`: A dictionary mapping active tokens (those with non-zero weights for this instance) to their weights.
    vector_tokens = dict(zip(tokens, weights))

    tokens_weights = {} # Stores individual weights of all q-grams and words found in the current tweet
    min_weight, max_weight = 0.00, 0.00 # For scaling colors of word highlights

    # `solo_palabras`: Aggregates weights for each "word" in the tweet.
    # A "word" here is a token from `tweet_tokens_one_words`.
    # Its aggregated weight is the sum of weights of all its constituent q-grams/tokens
    # (as per `app_data['text_model']`) that are present in `vector_tokens`.
    solo_palabras = {}

    # Iterate over each word from the simplified tokenization of the tweet
    for word in tweet_tokens_one_words:
        arreglo_de_pesos_per_word = np.array([])
        # Further tokenize this word using the main text model (which includes q-grams)
        for token in app_data['text_model'].tokenize(word):
            try:
                # If this sub-token (q-gram or word itself) has a weight, record it
                weight = vector_tokens[token]
                tokens_weights[token] = weight # Store individual token weight
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

    print(tweet_idx, distance, min_weight, max_weight) # For debugging, shows instance details and weight ranges.

    # Maps each word (from simplified tokenization) to its aggregated weight and corresponding color.
    words_weights = {key: [value,  value_to_color(value, min_weight, max_weight)] for key, value in solo_palabras.items()}

    # Determine the color for the overall decision value of the tweet
    decision_color_html = value_to_color(distance, app_data['min_distance_viz'], app_data['max_distance_viz'])

    return words_weights, tokens_weights, decision_color_html

def htmlizer():
    """
    Generates HTML table rows for visualizing tweet classifications, token weights,
    and prediction details. It iterates through the test set, filters by decision value,
    and calls `tracing_grams` to get token-level details.

    This function initializes `app_data['text_model_one_token']` if not already set.

    Returns:
        list: A list of HTML strings, where each string is a '<tr>...</tr>' element
              representing a tweet and its analysis.
    """
    # visualize structure

    global app_data
    # Initialize a simpler MicroTC model for tokenizing into words (primarily for display iteration)
    # Stores it in app_data to avoid re-initialization on every call if htmlizer is called multiple times.
    if 'text_model_one_token' not in app_data:
        app_data['text_model_one_token'] = microtc.TextModel(token_list=[-1], del_punc=True, del_diac=True, lc=True, usr_option='group')

    tr_list = []  # Initialize as list

    for idx, row in app_data['X_test'].iterrows():

        #if idx < 1020: # Example of a debugging skip, can be removed
        #    continue

        distance = row['distance'] # Overall decision score for this instance

        # Filter instances by the minimum decision value threshold
        if abs(distance) >= app_data['min_decision_value']:

            tweet = row['text']
            true_klass = row['klass']
            pred_klass = row['pred']

            # words_weights has the accumulated weights per word and it is what is displayed in the tweet
            # tokens_weights has all the unit weights per token (q_grams and words) and is the tooltip
            words_weights, tokens_weights, decision_color_html = tracing_grams(idx)

            tr_row = '' # Accumulates HTML for words in the current tweet

            # `unique_words_weights` is used to avoid displaying the same word-weight combination multiple times
            # if a word appears identically multiple times and contributes the same weight each time.
            # This typically happens if `words_weights` (from `tracing_grams`) could have duplicates if not handled there.
            unique_words_weights = {}
            seen_combinations = set()

            for word, (weight, color) in words_weights.items():
                if (word, weight) not in seen_combinations:
                    seen_combinations.add((word, weight))
                    unique_words_weights[word] = (weight, color)

            # Construct HTML for each word and its tooltip
            for word, (weight, color) in unique_words_weights.items():
                if abs(weight) >= app_data['min_weight']: # Apply min_weight threshold for highlighting
                    # `frag_tip`: Tooltip HTML for an individual highlighted word, showing breakdown by its q-grams.
                    tkn_frag_aggs = {} # Stores q-grams and their weights for the current word
                    for tkn_frag in app_data['text_model'].tokenize(word): # Tokenize word by main model
                        weight_frag = tokens_weights.get(tkn_frag, None)
                        if weight_frag:
                            tkn_frag_aggs[tkn_frag] = weight_frag

                    frag_tip = """<table class='tooltip_text'> """
                    frag_tip += f"<tr><td><strong>{word}</strong></td><td><strong/>∑{round(weight, 5):+.5f}</strong></td></tr>"
                    for g, v in tkn_frag_aggs.items():
                        # g is a q-gram (e.g., "q:ing") or a word token
                        if g.startswith('q:'):
                            frag_tip += f"<tr><td>{g.replace('q:', '')}</td><td>{round(v, 5):+.5f}</td></tr>"
                    frag_tip += "</table>"

                    # HTML span for the highlighted word with its tooltip
                    span_word = (f"<span class='sp_word tooltip' style='background-color:{color}; '>{word} "
                                 f"{frag_tip}</span>")

                else:
                    # Words with weights below the app_data['min_weight'] threshold are displayed plainly.
                    span_word = f"<span class='sp_word' style='color: unset;'>{word}</span>"
                tr_row += span_word

            weight_total = sum(tokens_weights.values()) # Total weight from all tokens contributing to this instance for the predicted class.
            # The line below is a note: if uncommented, it would filter tokens for the tooltip based on app_data['min_weight'].
            # tokens_weights_tooltip_filtered = {key: value for key, value in tokens_weights.items() if abs(value) >= app_data['min_weight']}
            weight_umbral = sum(tokens_weights.values()) # Currently, this is the same as weight_total as tooltip tokens are not filtered by min_weight for this sum.

            # Sort all tokens (q-grams and words) by absolute weight for display in the main prediction tooltip, and limit to top N.
            tokens_weights_sorted = dict(sorted(tokens_weights.items(), key=lambda item: abs(item[1]), reverse=True))
            num_de_qgrams = 15 # Max number of q-grams to show in the main tooltip
            if len(tokens_weights_sorted) > num_de_qgrams:
                primeros_grams = dict(list(tokens_weights_sorted.items())[:num_de_qgrams])
                restantes = list(tokens_weights_sorted.items())[num_de_qgrams:]
                peso_restante = sum([v for k, v in restantes])
                primeros_grams[f'{len(restantes)} more...'] = peso_restante
                tokens_weights_display = primeros_grams
            else:
                tokens_weights_display = tokens_weights_sorted

            # `span_tip`: Tooltip HTML for the overall prediction, showing top contributing tokens (q-grams/words).
            span_tip = """<table class='tooltip_text'> """
            span_tip += f"<tr><td><strong>Weight</strong></td><td><strong/>{round(weight_total, 5):+.5f}</strong></td></tr>" # Total weight before any display filtering
            span_tip += f"<tr><td><strong>Threshold</strong></td><td><strong/>{round(weight_umbral, 5):+.5f}</strong></td></tr>" # Total weight of displayed items (potentially after min_weight filter if it were applied above)
            for g, v in tokens_weights_display.items():
                # g is a q-gram or word token
                if g.startswith('q:'):
                    span_tip += f"<tr><td>{g.replace('q:', '')}</td><td>{round(v, 4):+.4f}</td></tr>"
                else: # Emphasize word tokens
                    span_tip += f"<tr><td style='font-weight: 600;'><em>{g}</em></td><td>{round(v, 4):+.4f}</td></tr>"
            span_tip += "</table>"

            # Visual indicator for correct (✔) or incorrect (✘) prediction
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

    return tr_list

def generate_chunks():
    """
    Generates HTML table row data in chunks for streaming to the client.
    Calls `htmlizer` to get all table rows and then yields them in batches.
    This is used for server-sent events (SSE) to update the UI progressively.

    Yields:
        str: Server-sent event data strings. Starts with '__START__', ends with '__COMPLETE__',
             and intermediate events are chunks of HTML table rows.
    """
    table_rows = htmlizer() # Get all HTML table rows

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
    """
    Streams HTML table row data to the client using Server-Sent Events (SSE).
    This allows the web page to update dynamically as data is processed.

    Returns:
        Response: A Flask Response object that streams data from `generate_chunks`.
    """
    return Response(stream_with_context(generate_chunks()), content_type='text/event-stream')

@app.route('/download-json')
def download_json():
    """
    Generates a JSON file containing detailed token weight information for each
    test instance that meets the `min_decision_value` threshold.
    The JSON structure includes prediction details and token weights for each instance.

    Returns:
        Response: A Flask response to send the generated JSON file for download.
    """
    def json_serializer(obj):
        """Custom JSON serializer to handle numpy numeric types."""
        if isinstance(obj, (np.integer, np.floating)):
            return str(obj)  # Convert numeric types to string
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    tokens_data = [] # List to hold data for each tweet

    # Iterate through test instances that meet the decision value threshold
    for idx, row in app_data['X_test'].iterrows():
        distance = row['distance']
        if abs(distance) >= app_data['min_decision_value']:
            tweet = row['text']
            true_klass = row['klass']
            pred_klass = row['pred']

            # Get token weights and other details for the current instance
            words_weights, tokens_weights, decision_color_html = tracing_grams(idx)

            # Structure for each entry in the JSON file:
            # { "instance_index": { "prediction_klass": "...", "decision_value": "...", "true_klass": "...", "data": {token: weight, ...} } }
            tokens_data.append({
                str(idx): {
                    'prediction_klass': str(pred_klass),
                    'decision_value': str(distance),
                    'true_klass': true_klass,
                    'data': tokens_weights # Dictionary of token: weight
                }
            })

    filename = 'data.json' # Default filename for download

    # Save the list of dictionaries to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(tokens_data, json_file, default=json_serializer, indent=4)

    # Submit the file for download
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)