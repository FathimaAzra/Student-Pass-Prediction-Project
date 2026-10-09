
const form = document.getElementById("predictionForm");
const resultContent = document.getElementById("resultContent");
const predictButton = document.getElementById("predictButton");

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    const student = {
        Study_Hours: Number(document.getElementById("studyHours").value),
        Attendance: Number(document.getElementById("attendance").value),
        Previous_Marks: Number(document.getElementById("previousMarks").value),
        Assignments: Number(document.getElementById("assignments").value)
    };

    // Validate values before sending them to Flask
    if (
        !Number.isFinite(student.Study_Hours) ||
        student.Study_Hours < 0 || student.Study_Hours > 24 ||
        !Number.isFinite(student.Attendance) ||
        student.Attendance < 0 || student.Attendance > 100 ||
        !Number.isFinite(student.Previous_Marks) ||
        student.Previous_Marks < 0 || student.Previous_Marks > 100 ||
        !Number.isInteger(student.Assignments) ||
        student.Assignments < 0 || student.Assignments > 10
    ) {
        resultContent.innerHTML = `
            <h2>Check your input</h2>
            <p>Enter values within the allowed ranges for all four fields.</p>
        `;
        return;
    }

    predictButton.disabled = true;
    predictButton.textContent = "Analyzing student data...";

    resultContent.innerHTML = `
        <div class="empty-illustration">…</div>
        <h2>Analyzing academic data</h2>
        <p>Please wait while the trained model generates its prediction.</p>
    `;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(student)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        const isPass = data.prediction === "Pass";
        const resultClass = isPass ? "pass" : "fail";
        const symbol = isPass ? "✓" : "!";
        const probability = Number(data.probability);

        if (!Number.isFinite(probability) ||
            probability < 0 || probability > 100) {
            throw new Error("The server returned an invalid probability.");
        }

        resultContent.innerHTML = `
            <div class="empty-illustration">${symbol}</div>
            <p class="prediction-label">PREDICTED OUTCOME</p>
            <h2 class="prediction-value ${resultClass}">
                ${isPass ? "Likely to Pass" : "Likely to Fail"}
            </h2>
            <p>
                The trained model predicts:
                <strong>${isPass ? "Pass" : "Fail"}</strong>
            </p>
            <div class="probability-track">
                <div class="probability-fill"
                     style="width: ${probability}%"></div>
            </div>
            <p class="probability-text">
                Model probability for predicted class: ${probability.toFixed(1)}%
            </p>
            <p>
                This is a model estimate, not a guarantee of the actual result.
            </p>
        `;
    } catch (error) {
        resultContent.innerHTML = `
            <h2>Unable to predict</h2>
            <p>${error.message}</p>
            <p>Check that the Flask server is running and try again.</p>
        `;
    } finally {
        predictButton.disabled = false;
        predictButton.innerHTML =
            'Predict student result <span>→</span>';
    }
});