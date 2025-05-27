let eventSource;

function showProgress() {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.display = 'block';
}

function hideProgress() {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.display = 'none';
}

document.getElementById('datasetForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const header_html = document.getElementById('header_html');
    header_html.innerHTML = 'Processing source dataset...';

    const threshold_html = document.getElementById('threshold_html');
    threshold_html.style.display = "none";

    const traceability_html = document.getElementById('traceability_html');
    traceability_html.style.display = "none";


    document.getElementById("table_body_tweet").innerText = '';

    const progressBar = document.getElementById('progress-bar');
    progressBar.style.display = 'block';

    showProgress();

    const formData = new FormData(this);

    fetch('/build_header', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        if (data.processed) {

            header_html.innerHTML = `
                <p>${data.message}</p>
                <table style="font-size: 12px; border-spacing: 2px;">
                    <tr>
                        <td class="label">X_train shape</td>
                        <td class="value">${data.X_train_shape}</td>
                        <td class="label">f1-score</td>
                        <td class="value">${data.f1}</td>
                    </tr>
                    <tr>
                        <td class="label">X_train_tokenized_shape</td>
                        <td class="value">${data.X_train_tokenized_shape}</td>
                        <td class="label">precision</td>
                        <td class="value">${data.precision}</td>
                    </tr>
                    <tr>
                        <td class="label">X_train_classes</td>
                        <td class="value">${data.X_train_klasses}</td>
                        <td class="label">recall</td>
                        <td class="value">${data.recall}</td>
                    </tr>
                    <tr>
                        <td class="label"></td>
                        <td class="value"></td>
                        <td class="label">accuracy</td>
                        <td class="value">${data.accuracy}</td>
                    </tr>                    
                    <tr>
                        <td class="label">X_test shape</td>
                        <td class="value">${data.X_test_shape}</td>
                        <td class="label"></td>
                        <td class="value"></td>
                    </tr>
                    <tr>
                        <td class="label">X_test_tokenized_shape</td>
                        <td class="value">${data.X_test_tokenized_shape}</td>
                        <td class="label"></td>
                        <td class="value"></td>
                    </tr>
                    <tr>
                        <td class="label">X_test_classes</td>
                        <td class="value">${data.X_test_klasses}</td>
                        <td class="label"></td>
                        <td class="value"></td>
                    </tr>
                    <tr><td class="label"></td><td class="value"></td><td class="label"></td><td class="value"></td></tr>
                    <tr>
                        <td class="label">Q-grams Pattern</td>
                        <td class="value">[-1, 2, 3, 4]</td>
                        <td class="label">density / variance</td>
                        <td class="value">${data.density}</td>
                    </tr>
                </table>
            `;

            threshold_html.style.display = "block";

            document.getElementById("table_body_tweet").innerText = '';

            // fetchData();

        } else {
            header_html.innerHTML =  `<p>${data.message}</p>`;
            threshold_html.style.display = "none";
        }

        /* hideProgress(); */

    })
    .catch(error => {
        console.error('Error:', error)
        threshold_html.style.display = "none";
        hideProgress();
    }).finally(() =>{
        hideProgress();
    });
});

document.getElementById('thresholdForm').addEventListener('submit', function(event) {
    event.preventDefault();

    document.getElementById("table_body_tweet").innerText = 'Processing target...';

    document.getElementById('msg_status').textContent = 'Processing target...';

    const header_threshol_html = document.getElementById('header_threshold_html');
    header_threshol_html.innerHTML = 'Processing target...';

    const traceability_html = document.getElementById('traceability_html');
    traceability_html.style.display = "none";

    const progressBar = document.getElementById('progress-bar');
    progressBar.style.display = 'block';

    showProgress();

    const thresholdForm = new FormData(this);

    fetch('/threshold_header', {
        method: 'POST',
        body: thresholdForm
    })
    .then(response => response.json())
    .then(data => {

        if (data.processed) {

            header_threshol_html.innerHTML = `
                <p>${data.message}</p>
                <table style="font-size: 12px; border-spacing: 2px;">
                    <tr>
                        <td class="label">f1-score</td>
                        <td class="value">${data.f1}</td>
                        <td class="value">${data.f1_diff > 0 ? "+" : ""}${data.f1_diff.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td class="label">precision</td>
                        <td class="value">${data.precision}</td>
                        <td class="value">${data.precision_diff > 0 ? "+" : ""}${data.precision_diff.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td class="label">recall</td>
                        <td class="value">${data.recall}</td>
                        <td class="value">${data.recall_diff > 0 ? "+" : ""}${data.recall_diff.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td class="label">accuracy</td>
                        <td class="value">${data.accuracy}</td>
                        <td class="value">${data.accuracy_diff > 0 ? "+" : ""}${data.accuracy_diff.toFixed(2)}%</td>
                    </tr>
                    <tr>
                        <td class="label">density / variance</td>
                        <td class="value">${data.density}</td>
                        <td class="value"></td>
                    </tr>
                    <tr>
                        <td class="label">decision function = w⋅x + b </td>
                        <td class="value">Intercept (b)</td>
                        <td class="value">${data.intercept_ > 0 ? "+" : ""}${data.intercept_}</td>
                    </tr>
                </table>`;

            document.getElementById("table_body_tweet").innerText = '';

            traceability_html.style.display = "block";

            // fetchData();
        } else {
            document.getElementById("table_body_tweet").innerText = '<p>${data.message}</p>';
            traceability_html.style.display = "none";
        }
    })
    .catch(error => {
        console.error('Error:', error)
        document.getElementById('msg_status').textContent = 'Error in processing target';
        traceability_html.style.display = "none";
        hideProgress();
    }).finally(() =>{
        document.getElementById('msg_status').textContent = 'Ready';
        hideProgress();
    });

});

document.getElementById('traceabilityForm').addEventListener('submit', function(event) {
    event.preventDefault();

    document.getElementById("table_body_tweet").innerText = '';
    document.getElementById('msg_status').textContent = 'Tracing q-grams...';

    fetchData()
});

function fetchData() {

    if (!eventSource || (eventSource.readyState === EventSource.CLOSED)) {
        eventSource = new EventSource("/stream-data-html");

        eventSource.onmessage = function (event) {
            let content = event.data
            if (content === "__COMPLETE__") {
                document.getElementById('msg_status').textContent = "All elements have been loaded";
                eventSource.close();
            } else if (content === '__START__') {

                table_header = "<tr><th>Idx</th><th>Tweet: words highlighted their contribution on w &middot; p</th>" +
                    "<th>" +
                    "   <span class='tooltip' style='float: left;'>" +
                            "<span class='span_pred_value' >Decision</span>" +
                            "<span class='span_pred_klass' >P</span>" +
                            "<span class='span_true_klass' >T</span>" +
                            "<span class='tooltip_text' style='width: 160px; text-align: left; text-shadow: 0 0 BLACK;'><ul >" +
                                "<li>Decision value</li>" +
                                "<li>Prediction class</li>" +
                                "<li>True class</li></ul>" +
                            "</span>" +
                        "</span>" +
                    "</th>" +
                    "</tr>"

                document.getElementById("table_body_tweet").insertAdjacentHTML('beforeend', table_header);

                document.getElementById('msg_status').textContent = 'Loading...';

            } else {
                /* each row tr */
                document.getElementById("table_body_tweet").insertAdjacentHTML('beforeend', content);
            }
        };

        eventSource.onerror = function() {
            document.getElementById('msg_status').textContent = "Error en la conexión.";
            eventSource.close();
        };

    }

}
