const labels = [
  "Baseball",
  "Boy-Girl",
  "C_Chat",
  "HatePolitics",
  "Lifesmoney",
  "Military",
  "PC_Shopping",
  "Stock",
  "Tech_Job",
];

document.getElementById("predict-btn").onclick = async () => {
  const title = document.getElementById("title-input").value.trim();
  if (!title) return alert("請輸入標題");

  const res = await fetch(
    `/api/model/prediction?title=${encodeURIComponent(title)}`
  );
  const data = await res.json();

  // 顯示結果
  const predLabel = data.label;
  document.getElementById("pred-label").textContent = predLabel;
  document.getElementById(
    "pred-link"
  ).href = `https://www.ptt.cc/bbs/${predLabel}/index.html`;
  document.getElementById("result-block").style.display = "block";

  // 產生建議按鈕
  const sugDiv = document.getElementById("suggestions");
  sugDiv.innerHTML = "";
  labels.forEach((lab) => {
    const btn = document.createElement("button");
    btn.textContent = lab;
    btn.onclick = () => sendFeedback(title, lab);
    sugDiv.appendChild(btn);
  });
};

async function sendFeedback(title, label) {
  await fetch("/api/model/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ title, label }),
  });
  alert("感謝回饋！");
}
