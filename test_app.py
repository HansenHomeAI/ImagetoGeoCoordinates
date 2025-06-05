from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/upload', methods=['POST'])
def upload_test():
    return jsonify({'message': 'Test upload endpoint working'})

if __name__ == '__main__':
    print("Starting test server...")
    app.run(debug=True, host='0.0.0.0', port=8080) 