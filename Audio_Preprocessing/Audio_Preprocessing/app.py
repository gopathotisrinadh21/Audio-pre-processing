from flask import Flask, request,render_template
import numpy as np
import soundfile as sf

app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the file from the POST request
    file = request.files['file']

    # Load audio signal
    audio, samplerate = sf.read(file)

    # Convert audio to matrix form
    A = np.array(audio).T

    # Compute SVD of A
    U, S, VT = np.linalg.svd(A, full_matrices=False)

    # Truncate the singular values
    n_components = 100  # Number of components to keep
    S_trunc = np.zeros_like(S)
    S_trunc[:n_components] = S[:n_components]

    # Reconstruct the signal
    A_trunc = np.dot(U, np.dot(np.diag(S_trunc), VT))
    audio_trunc = A_trunc.T

    # Save the truncated audio signal
    sf.write('static/audio_trunc.wav', audio_trunc, samplerate)

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
