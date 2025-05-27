let eventSource;

function showProgress() {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.display = 'block';
}

function hideProgress() {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.display = 'none';
}

// Handles the submission of the dataset form.
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

            // Successfully processed the dataset
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

        } else {
            // Display error message if dataset processing failed
            header_html.innerHTML =  `<p>${data.message}</p>`;
            threshold_html.style.display = "none";
        }

    })
    .catch(error => {
        // Handle network or other errors during dataset processing
        console.error('Error:', error);
        header_html.innerHTML = '<p class="error-message">Error processing dataset. Please check the files and try again.</p>';
        threshold_html.style.display = "none";
        hideProgress(); // Ensure progress is hidden on error
    }).finally(() =>{
        hideProgress(); // Always hide progress after fetch attempt
    });
});

// Handles the submission of the threshold form.
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

            // Successfully processed thresholds
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

        } else {
            // Display error message if threshold processing failed
            document.getElementById("table_body_tweet").innerText = `<p>${data.message}</p>`;
            traceability_html.style.display = "none";
        }
    })
    .catch(error => {
        // Handle network or other errors during threshold processing
        console.error('Error:', error);
        document.getElementById('msg_status').textContent = 'Error in processing target';
        traceability_html.style.display = "none";
        hideProgress(); // Ensure progress is hidden on error
    }).finally(() =>{
        document.getElementById('msg_status').textContent = 'Ready';
        hideProgress(); // Always hide progress after fetch attempt
    });

});

// Initiates the data streaming process for traceability.
document.getElementById('traceabilityForm').addEventListener('submit', function(event) {
    event.preventDefault();

    document.getElementById("table_body_tweet").innerText = '';
    document.getElementById('msg_status').textContent = 'Tracing q-grams...';

    fetchData();
});

// Manages Server-Sent Events (SSE) for streaming HTML data to update the table.
function fetchData() {

    if (!eventSource || (eventSource.readyState === EventSource.CLOSED)) {
        eventSource = new EventSource("/stream-data-html");

        eventSource.onmessage = function (event) {
            let content = event.data;
            if (content === "__COMPLETE__") {
                // Handle stream completion
                document.getElementById('msg_status').textContent = "All elements have been loaded";
                eventSource.close();
            } else if (content === '__START__') {
                // Handle stream start: insert table header
                table_header = "<tr><th>Idx</th><th>Tweet: words highlighted their contribution on w &middot; p</th>" +
                    "<th>" +
                    // Apply CSS classes instead of inline styles
                    "   <span class='tooltip tooltip-header-float'>" +
                            "<span class='span_pred_value' >Decision</span>" +
                            "<span class='span_pred_klass' >P</span>" +
                            "<span class='span_true_klass' >T</span>" +
                            // Apply CSS classes instead of inline styles
                            "<span class='tooltip_text tooltip-custom-style'><ul >" +
                                "<li>Decision value</li>" +
                                "<li>Prediction class</li>" +
                                "<li>True class</li></ul>" +
                            "</span>" +
                        "</span>" +
                    "</th>" +
                    "</tr>";

                document.getElementById("table_body_tweet").insertAdjacentHTML('beforeend', table_header);
                document.getElementById('msg_status').textContent = 'Loading...';

            } else {
                // Handle incoming data chunks (table rows)
                document.getElementById("table_body_tweet").insertAdjacentHTML('beforeend', content);
            }
        };

        eventSource.onerror = function() {
            // Handle SSE connection errors
            document.getElementById('msg_status').textContent = "Error en la conexión.";
            eventSource.close();
        };
    }
}