// document.getElementById("predictForm").addEventListener("submit", async function(e) {
//     e.preventDefault();

//     // Collect input values
//     const data = {
//         age: parseInt(document.getElementById("age").value),
//         systolic_bp: parseFloat(document.getElementById("systolicBP").value),
//         diastolic_bp: parseFloat(document.getElementById("diastolicBP").value),
//         blood_sugar: parseFloat(document.getElementById("bloodSugar").value),
//         heart_rate: parseInt(document.getElementById("heartRate").value),
//         body_temp: parseFloat(document.getElementById("bodyTemp").value),
//         language : document.getElementById("language").value
//     };

//     try {
//         // Send request to backend
//         const response = await fetch("http://127.0.0.1:8000/predict", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify(data)
//         });

//         if (!response.ok) throw new Error("Network response was not ok");

//         const result = await response.json();
//         const resultDiv = document.getElementById("result");
//         resultDiv.textContent = `Predicted Risk: ${result.prediction}`;

//         // Color coding based on risk
//         if (result.prediction.toLowerCase().includes("low")) {
//             resultDiv.style.backgroundColor = "#4CAF50"; // green
//         } else if (result.prediction.toLowerCase().includes("medium")) {
//             resultDiv.style.backgroundColor = "#FF9800"; // orange
//         } else if (result.prediction.toLowerCase().includes("high")) {
//             resultDiv.style.backgroundColor = "#F44336"; // red
//         } else {
//             resultDiv.style.backgroundColor = "#607D8B"; // grey
//         }

//     } catch (error) {
//         const resultDiv = document.getElementById("result");
//         resultDiv.textContent = `Error: ${error.message}`;
//         resultDiv.style.backgroundColor = "#F44336";
//         console.error("Error:", error);
//     }
// });

document.getElementById("predictForm").addEventListener("submit", async function(e){
    e.preventDefault();

    const data = {
        age: parseInt(document.getElementById("age").value),
        systolic_bp: parseFloat(document.getElementById("systolicBP").value),
        diastolic_bp: parseFloat(document.getElementById("diastolicBP").value),
        blood_sugar: parseFloat(document.getElementById("bloodSugar").value),
        heart_rate: parseInt(document.getElementById("heartRate").value),
        body_temp: parseFloat(document.getElementById("bodyTemp").value),
        language: document.getElementById("language").value
    };

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await res.json();

        const resultDiv = document.getElementById("result");
        const sentencesDiv = document.getElementById("sentences");

        resultDiv.innerText = `Prediction: ${result.prediction}`;
        sentencesDiv.innerText = result.explanation;

        // Set dynamic colors
        if(result.prediction === "High Risk") resultDiv.style.color = "red";
        else if(result.prediction === "Medium Risk") resultDiv.style.color = "orange";
        else resultDiv.style.color = "green"; // Low Risk

    } catch (err) {
        console.error("Error:", err);
        document.getElementById("result").innerText = "Error occurred. Check console.";
        document.getElementById("result").style.color = "#fff";
        document.getElementById("sentences").innerText = "";
    }
});