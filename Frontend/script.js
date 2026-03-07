async function handleSubmit() {

    // Collect values
    const language   = document.getElementById("language").value;
    const age        = parseInt(document.getElementById("age").value);
    const systolicBP = parseFloat(document.getElementById("systolicBP").value);
    const diastolicBP= parseFloat(document.getElementById("diastolicBP").value);
    const bloodSugar = parseFloat(document.getElementById("bloodSugar").value);
    const heartRate  = parseInt(document.getElementById("heartRate").value);
    const bodyTemp   = parseFloat(document.getElementById("bodyTemp").value);

    // Basic validation
    if (!language) {
        alert("Please select a language.");
        return;
    }

    const data = {
        age:          age,
        systolic_bp:  systolicBP,
        diastolic_bp: diastolicBP,
        blood_sugar:  bloodSugar,
        heart_rate:   heartRate,
        body_temp:    bodyTemp,
        language:     language
    };

    // ── UI: loading state ──
    const btn       = document.getElementById("submitBtn");
    const loader    = document.getElementById("btnLoader");
    const btnText   = btn.querySelector(".btn-text");
    const btnIcon   = btn.querySelector(".btn-icon");

    btn.disabled    = true;
    loader.classList.add("active");
    btnText.textContent = "Analysing…";
    btnIcon.textContent = "⏳";

    // Hide previous results / errors
    document.getElementById("emptyState").style.display    = "none";
    document.getElementById("resultsContent").style.display= "none";
    document.getElementById("errorState").style.display    = "none";

    try {
        // ── Core API call (unchanged) ──
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(data)
        });

        const result = await res.json();

        // ── Risk badge ──
        const riskBadge = document.getElementById("riskBadge");
        const riskLabel = document.getElementById("riskLabel");
        const riskIcon  = document.getElementById("riskIcon");
        const riskSub   = document.getElementById("riskSub");

        riskBadge.className = "risk-banner";

        if (result.prediction === "High Risk") {
            riskBadge.classList.add("high");
            riskIcon.textContent = "🚨";
            riskSub.textContent  = "Immediate medical attention advised.";
        } else if (result.prediction === "Medium Risk") {
            riskBadge.classList.add("med");
            riskIcon.textContent = "⚠️";
            riskSub.textContent  = "Some vitals need monitoring.";
        } else {
            riskBadge.classList.add("low");
            riskIcon.textContent = "✅";
            riskSub.textContent  = "Vitals are within acceptable range.";
        }

        riskLabel.textContent = result.prediction;

        // ── Vitals summary chips ──
        const chips = [
            { label: "Age",        value: `${age} yrs`           },
            { label: "BP",         value: `${systolicBP}/${diastolicBP}` },
            { label: "Blood Sugar",value: `${bloodSugar} mmol/L`  },
            { label: "Temp",       value: `${bodyTemp}°F`         },
            { label: "Heart Rate", value: `${heartRate} bpm`      },
            { label: "Language",   value: language.slice(0, 5)    },
        ];

        const summary = document.getElementById("vitalsSummary");
        summary.innerHTML = chips.map(c => `
            <div class="vs-chip">
                <span class="vs-chip-label">${c.label}</span>
                <span class="vs-chip-value">${c.value}</span>
            </div>
        `).join("");

        // ── AI Explanation — always English ──
        document.getElementById("explanationBox").textContent = result.explanation;

        // ── Regional translation box ──
        const transSection = document.getElementById("translationSection");
        if (language !== "English" && result.translated_explanation) {
            document.getElementById("translationTitle").textContent =
                `🗣️ Explanation in ${language}`;
            document.getElementById("translationBox").textContent = result.translated_explanation;
            transSection.style.display = "block";
        } else {
            transSection.style.display = "none";
        }

        // Show results
        document.getElementById("resultsContent").style.display = "flex";

    } catch (err) {
        console.error("Error:", err);
        document.getElementById("errorState").style.display = "flex";

    } finally {
        // ── UI: reset button ──
        btn.disabled = false;
        loader.classList.remove("active");
        btnText.textContent = "Analyse Risk";
        btnIcon.textContent = "🔍";
    }
}