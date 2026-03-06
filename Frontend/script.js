document.getElementById("riskForm").addEventListener("submit", function(e){

e.preventDefault()

let sbp = document.getElementById("sbp").value
let hb = document.getElementById("hb").value

let risk = ""
let advice = ""

if(sbp > 140 || hb < 8){

risk = "HIGH RISK"
advice = "Refer patient to PHC within 24 hours."

}

else if(sbp > 120){

risk = "MEDIUM RISK"
advice = "Monitor vitals twice a week."

}

else{

risk = "LOW RISK"
advice = "Normal pregnancy monitoring."

}

let result = document.getElementById("result")
let level = document.getElementById("riskLevel")
let rec = document.getElementById("recommendation")

result.classList.remove("hidden")

level.innerHTML = risk

if(risk === "HIGH RISK")
level.className="high"

else if(risk === "MEDIUM RISK")
level.className="medium"

else
level.className="low"

rec.innerHTML = advice

})