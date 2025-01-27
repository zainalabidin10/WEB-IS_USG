<?php

use App\Http\Controllers\tambah_penyakit;
use App\Http\Controllers\SheepController;
use Illuminate\Support\Facades\Route;
use Spatie\FlareClient\View;

Route::get('/', function () {
    return view('index');
});

Route::get('/create-sheep', function () {
    return view('createSheep');
})->name('sheep.create');

Route::get('/read-sheep', function () {
    return view('readSheep');
})->name('sheep.read');

Route::get('/edit-sheep', function () {
    return view('editSheep');
})->name('sheep.edit');

Route::get('/create-vitalSign', function () {
    return view('createVitalSign');
})->name('vitalSign.create');

Route::get('/create-Radiologi', function () {
    return view('createRadiologi');
})->name('Radiologi.create');

Route::resource('sheep', SheepController::class);

// Route::get('/tambah-penyakit', function () {
//     return view('tambah-penyakit');
// });
Route::get('/list-penyakit', [tambah_penyakit::class,'index_penyakit'])->name('index_penyakit');

Route::prefix('/list-penyakit')->group(function(){
    Route::get('/form-tambah', [tambah_penyakit::class, 'form_tambah'])->name('form_tambah');
    Route::post('/form-tambah/simpan',[tambah_penyakit::class,'simpan'])->name('simpan_data');
    Route::post('/hapus-data', [tambah_penyakit::class, 'delete'])->name('hapus_data');
    Route::get('/form-edit/{id}', [tambah_penyakit::class, 'form_edit'])->name('form_edit');
    Route::post('/form-edit/{id}/edit',[tambah_penyakit::class,'update'])->name('edit_data');
});
