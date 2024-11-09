document.getElementById('symptomForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission
    
    // Get the symptom input value
    const symptomInput = document.getElementById('symptomInput').value;
    
    // Prepare data to send
    const data = {
        symptoms: symptomInput.split(',').map(symptom => symptom.trim())  // Split input by comma and trim spaces
    };
    
    // Send data to the server using Fetch API
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Display the result in the result div
        const resultDiv = document.getElementById('result');
        const diseaseList = document.getElementById('diseaseList');
        diseaseList.innerHTML = ''; // Clear previous results
        result.predictions.forEach(([disease, probability]) => {
            const li = document.createElement('li');
            li.innerText = `${disease}: ${(probability * 100).toFixed(2)}%`;
            diseaseList.appendChild(li);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred while processing your request.';
    });
});
