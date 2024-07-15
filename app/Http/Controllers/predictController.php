<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class predictController extends Controller
{
    public function predict(Request $request)
    {
        $input = $request->input('data');

        // Memanggil script Python untuk memprediksi
        $output = shell_exec("python3 " . storage_path('app/models/predict.py') . " " . escapeshellarg($input));

        return response()->json(['prediction' => $output]);
    }
}
