# Menggunakan base image Python
FROM python:3.8-slim

# Mengatur direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt ke dalam container
COPY requirements.txt requirements.txt

# Menginstall dependensi
RUN pip install --upgrade pip && pip install -r requirements.txt

# Menyalin seluruh kode aplikasi ke dalam container
COPY . .

# Mengekspos port 5000 (port default Flask)
EXPOSE 5000

# Menjalankan aplikasi Flask
CMD ["python", "app.py"]
