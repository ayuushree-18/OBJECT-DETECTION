const fileInput = document.getElementById('fileInput');
const sendBtn = document.getElementById('sendBtn');
const resultImg = document.getElementById('resultImg');
const jsonOut = document.getElementById('jsonOut');

sendBtn.onclick = async () => {
  if (!fileInput.files.length) {
    alert("Choose an image first");
    return;
  }
  const file = fileInput.files[0];
  const fd = new FormData();
  fd.append('file', file);
  sendBtn.disabled = true;
  sendBtn.textContent = "Uploading...";
  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    if (!resp.ok) {
      alert("Server error: " + resp.statusText);
      return;
    }
    const data = await resp.json();
    jsonOut.textContent = JSON.stringify(data.detections, null, 2);
    resultImg.src = "data:image/png;base64," + data.image_base64;
  } catch (e) {
    alert("Request failed: " + e.message);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }
};
