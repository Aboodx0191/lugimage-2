
function uploadImage() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    // Display image preview
    let reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById("imagePreview").innerHTML = `
            <img src="${e.target.result}" class="img-fluid" style="max-height: 300px;">
        `;
    };
    reader.readAsDataURL(file);

    let formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("Response data:", data);
        document.getElementById("result").innerHTML = `
            <div class="alert ${data.prediction === 'Pneumonia' ? 'alert-danger' : 'alert-success'}">
                <h3>Prediction: ${data.prediction}</h3>
                <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
            </div>
        `;
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = `
            <div class="alert alert-warning">
                <p>Error making prediction: ${error.message}</p>
            </div>
        `;
    });
}