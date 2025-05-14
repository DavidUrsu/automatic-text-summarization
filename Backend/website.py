
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Text Titler</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            font-family: sans-serif;
            background-color: #f5f5f5;
        }
        textarea, select, button {
            margin: 10px;
            width: 300px;
            font-size: 16px;
        }
        textarea {
            height: 100px;
        }
        #response {
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h2>Enter your text</h2>
    <p id="response"></p>
    <textarea id="inputText" placeholder="Paste your text here..."></textarea>
    <select id="modelSelect">
        <option value="bart">Bart</option>
        <option value="t5">T5</option>
        <option value="fine tunned">Fine Tunned</option>
        <option value="extractive">Extractive</option>
    </select>
    <button onclick="sendText()">Generate Title</button>
    

    <script>
        function sendText() {
            const text = document.getElementById("inputText").value;
            const model = document.getElementById("modelSelect").value;

            fetch("/generate_title", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ text: text, model: model })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById("response").textContent = data.title;
            });
        }
    </script>
</body>
</html>
"""