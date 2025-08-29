const input = document.getElementById("userInput");

// Listen for Enter key
input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault(); // prevent newline
        sendMessage();
    }
});

async function sendMessage() {
  const input = document.getElementById("userInput");
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  // Show "thinking..."
  const loadingMsg = addMessage("Bot is typing...", "bot");

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text })
  });
  const data = await response.json();

  // Replace loading with actual reply
  loadingMsg.textContent = data.reply;
}

function addMessage(text, sender) {
  const messages = document.getElementById("messages");
  const msgDiv = document.createElement("div");
  msgDiv.className = "message " + sender;
  msgDiv.textContent = text;
  messages.appendChild(msgDiv);
  messages.scrollTop = messages.scrollHeight;
  return msgDiv; // return ref so we can update it
}
