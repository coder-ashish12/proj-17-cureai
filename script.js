function predict() {
    let symptoms = document.getElementById("symptoms").value;
    const resultContainer = document.getElementById("result-container");
    const diseaseSpan = document.getElementById("disease");
    const precautionsList = document.getElementById("precautions");

    if (!symptoms) {
        alert("Please enter your symptoms!");
        return;
    }

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: symptoms })
    })
    .then(response => response.json())
    .then(data => {
        // Update the disease name in the span
        diseaseSpan.innerText = data.predicted_disease;

        // Clear and populate the precautions list
        precautionsList.innerHTML = "";
        data.precautions.forEach(precaution => {
            let li = document.createElement("li");
            li.innerText = precaution;
            precautionsList.appendChild(li);
        });

        // Show the result container
        resultContainer.style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        diseaseSpan.innerText = "Server Error! Check Flask API.";
        precautionsList.innerHTML = ""; // Clear precautions on error
        resultContainer.style.display = "block"; // Still show the container to display the error
    });
}