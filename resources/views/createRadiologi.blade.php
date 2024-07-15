@extends('layouts.template')
@section('content')
    <main id="main" class="main">
        <div class="row">
            <div class="card">
                <div class="card-body d-flex justify-content-between align-items-center p-2">
                    <div>daftar assesmen</div>
                </div>
            </div>
        </div>
        <div class="row">
            <form action="" method="post" enctype="multipart/form-data">
                @csrf
                <div class="card py-5">
                    <div class="container">
                        <div class="row">
                            <div class="col-md-8">
                                <div class="row">
                                    <div class="col-md-6 my-2">
                                        <label for="id_assesmen" class="form-label text-uppercase">ID asesmen</label>
                                        <input type="text" id="id_assesmen" placeholder="masukan ID ASESMEN"
                                            class="form-control" name="id_assesmen" required>
                                    </div>
                                    <div class="col-md-6 my-2">
                                        <label for="id_domba" class="form-label text-uppercase">ID domba</label>
                                        <input type="text" id="id_domba" placeholder="masukan ID domba"
                                            class="form-control" name="id_domba" required>
                                    </div>
                                    <div class="col-md-6 my-2">
                                        <label for="tanggal" class="form-label text-uppercase">tanggal assesmen</label>
                                        <input type="date" name="tanggal_assesmen" id="tanggal" placeholder="tanggal"
                                            class="form-control" required>
                                    </div>
                                    <div class="col-md-6 my-2">
                                        <label for="usia_domba" class="form-label text-uppercase">usia domba</label>
                                        <input type="text" name="usia_domba" id="usia_domba" placeholder="usia domba"
                                            class="form-control" required>
                                    </div>
                                    <div class="col-md-6 my-2">
                                        <label for="uploadGambar" class="form-label text-uppercase">Upload Gambar</label>
                                        <input type="file" name="uploadGambar" id="uploadGambar" class="form-control"
                                            accept=".jpg, .jpeg, .png" required>
                                        <img id="gambarUpload" alt="gambar upload"
                                            style="margin-top: 10px; max-width: 100%;">
                                    </div>
                                    <div class="col-md-6 my-2">
                                        <label for="status" class="form-label text-uppercase">Status</label>
                                        <input type="text" name="status" id="status" placeholder="Status"
                                            class="form-control" readonly>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div>
                                    <label for="assesor" class="form-label">nama assesor</label>
                                    <input type="text" id="assesor" placeholder="nama assesor" class="form-control"
                                        name="assesor" required>
                                </div>
                                <br/>
                                <div class="col-md-12">
                                <label for="keterangan" class="form-label">keterangan lain</label>
                                <div class="form-floating">
                                    <textarea class="form-control" placeholder="Leave a comment here" id="floatingTextarea2" style="height: 300px"
                                        class="form-control" name="keterangan"></textarea>
                                    <label for="floatingTextarea2">ketik disini</label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="container">
                        <div class="row d-flex justify-content-between">
                            <div class="col-md-5">
                                <div class="row">
                                    
                                </div>
                            </div>
                            
                            </div>
                        </div>
                    </div>
                </div>
                <div class="mx-auto" style="width: 300px;">
                    <button type="submit" class="btn btn-primary" style="width: 316px">tambah</button>
                </div>
            </form>
        </div>
    </main>

    <script>
        document.getElementById('uploadGambar').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('gambarUpload').src = e.target.result;
                };
                reader.readAsDataURL(file);

                // Upload image to Flask API and get prediction
                const formData = new FormData();
                formData.append('file', file);

                fetch('http://localhost:5000/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('status').value = data.prediction;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
            }
        });
    </script>
@endsection
