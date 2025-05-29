from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
import shutil
import zipfile
import io
from clustering import process_images

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
CLUSTERED_FOLDER = "clustered"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLUSTERED_FOLDER, exist_ok=True)

@app.route("/process", methods=["POST"])
def process_faces():
    user_id = request.args.get("user")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing user ID"})

    user_upload_dir = os.path.join(UPLOAD_FOLDER, user_id)
    user_cluster_dir = os.path.join(CLUSTERED_FOLDER, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    os.makedirs(user_cluster_dir, exist_ok=True)

    uploaded_files = request.files.getlist("images")
    paths = []
    for file in uploaded_files:
        path = os.path.join(user_upload_dir, file.filename)
        file.save(path)
        paths.append(path)

    try:
        process_images(paths, user_cluster_dir)
        return jsonify({"status": "success", "message": "Faces clustered"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/clusters", methods=["GET"])
def list_clusters():
    user_id = request.args.get("user")
    user_cluster_dir = os.path.join(CLUSTERED_FOLDER, user_id)
    if not os.path.exists(user_cluster_dir):
        return jsonify({})

    result = {}
    for folder_name in os.listdir(user_cluster_dir):
        folder_path = os.path.join(user_cluster_dir, folder_name)
        if os.path.isdir(folder_path):
            images = [img for img in os.listdir(folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                result[folder_name] = images
    return jsonify(result)

@app.route('/clustered/<user>/<person>/<image>')
def serve_image(user, person, image):
    return send_from_directory(os.path.join(CLUSTERED_FOLDER, user, person), image)

@app.route("/download_cluster", methods=["GET"])
def download_cluster():
    user_id = request.args.get("user")
    folder_name = request.args.get("folder")
    folder_path = os.path.join(CLUSTERED_FOLDER, user_id, folder_name)

    if not os.path.exists(folder_path):
        return jsonify({"status": "error", "message": "Folder not found"}), 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zf.write(full_path, arcname)
    memory_file.seek(0)

    return send_file(
        memory_file,
        download_name=f"{folder_name}.zip",
        as_attachment=True
    )

@app.route("/delete_all", methods=["POST"])
def delete_user_data():
    user_id = request.args.get("user")
    if not user_id:
        return jsonify({"status": "error", "message": "Missing user ID"})

    upload_path = os.path.join(UPLOAD_FOLDER, user_id)
    cluster_path = os.path.join(CLUSTERED_FOLDER, user_id)

    try:
        if os.path.exists(upload_path):
            shutil.rmtree(upload_path)
        if os.path.exists(cluster_path):
            shutil.rmtree(cluster_path)
        return jsonify({"status": "success", "message": f"Data deleted for user {user_id}"})
    except Exception as e:
        print(f"Delete failed: {e}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
