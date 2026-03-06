const resources = {

en:{
translation:{
subtitle:"AI Maternal Health Risk Assessment",
title:"Enter Patient Vitals",
age:"Age",
sbp:"Systolic BP",
dbp:"Diastolic BP",
bs:"Blood Sugar",
temp:"Body Temperature",
hr:"Heart Rate",
hb:"Hemoglobin",
predict:"Predict Risk",
result:"Risk Result"
}
},

hi:{
translation:{
subtitle:"एआई मातृ स्वास्थ्य जोखिम मूल्यांकन",
title:"रोगी के स्वास्थ्य विवरण दर्ज करें",
age:"आयु",
sbp:"सिस्टोलिक रक्तचाप",
dbp:"डायस्टोलिक रक्तचाप",
bs:"ब्लड शुगर",
temp:"शरीर का तापमान",
hr:"हृदय गति",
hb:"हीमोग्लोबिन",
predict:"जोखिम का अनुमान लगाएं",
result:"जोखिम परिणाम"
}
},

mr:{
translation:{
subtitle:"एआय मातृ आरोग्य जोखीम मूल्यांकन",
title:"रुग्णाची माहिती भरा",
age:"वय",
sbp:"सिस्टोलिक रक्तदाब",
dbp:"डायस्टोलिक रक्तदाब",
bs:"रक्तातील साखर",
temp:"शरीराचे तापमान",
hr:"हृदय गती",
hb:"हीमोग्लोबिन",
predict:"जोखीम तपासा",
result:"जोखीम परिणाम"
}
}

}

let lang = localStorage.getItem("language") || "en"

i18next.init({
lng: lang,
resources
}, function(err,t){

updateContent()

})

function updateContent(){

document.querySelectorAll("[data-i18n]").forEach(el=>{
el.innerHTML = i18next.t(el.getAttribute("data-i18n"))
})

document.querySelectorAll("[data-i18n-placeholder]").forEach(el=>{
el.placeholder = i18next.t(el.getAttribute("data-i18n-placeholder"))
})

}

/* Demo risk prediction */

document.getElementById("riskForm").addEventListener("submit",function(e){

e.preventDefault()

let sbp = document.getElementById("sbp").value
let hb = document.getElementById("hb").value

let risk=""
let advice=""

if(sbp>140 || hb<8){

risk="HIGH RISK"
advice="Refer patient to PHC immediately"

document.getElementById("riskLevel").className="risk-card high"

}

else if(sbp>120){

risk="MEDIUM RISK"
advice="Monitor vitals twice per week"

document.getElementById("riskLevel").className="risk-card medium"

}

else{

risk="LOW RISK"
advice="Continue normal monitoring"

document.getElementById("riskLevel").className="risk-card low"

}

document.getElementById("riskLevel").innerText=risk
document.getElementById("recommendation").innerText=advice

let ctx=document.getElementById("shapChart")

new Chart(ctx,{
type:'bar',
data:{
labels:["BP","Hemoglobin","Sugar","Heart Rate","Age"],
datasets:[{
label:"Feature Impact",
data:[40,30,15,10,5]
}]
}
})

})