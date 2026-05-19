/*
  Animal Sound Classification (Cow / Dog / Frog) on Arduino Nano 33 BLE Sense
  - Uses PDM mic input (16 kHz)
  - Computes MFCCs (13 x 200) on-device
  - Runs INT8 TFLite Micro model: input int8, output int8 (3 classes)
  - Prints predicted class + scores over Serial

  You MUST:
  1) Replace the model header include with your generated header (from animal_sound_int8.tflite)
  2) Ensure mel_filter_bank.h matches your Python MFCC settings (same SR, n_fft, hop, n_mels, fmin/fmax)
*/

#include <PDM.h>
#include <arduinoFFT.h>
#include <math.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// =======================
// CHANGE THESE INCLUDES
// =======================
#include "animal_sound_model.h"  // <-- your exported .h (int8 model array)
#include "mel_filter_bank.h"     // <-- mel filter bank array

// =======================
// Audio / MFCC parameters
// Must match Python training
// =======================
#define SAMPLE_RATE 16000

#define FRAME_SIZE 512
#define HOP_LENGTH 256
#define NUM_FRAMES 200

#define NUM_MFCC 13
#define NUM_MEL_FILTERS 26

// =======================
// Model parameters
// =======================
#define NUM_CLASSES 3  // Cow, Dog, Frog

// Audio buffer: stores one "clip" long enough for NUM_FRAMES frames
#define BUFFER_SIZE (FRAME_SIZE + (HOP_LENGTH * (NUM_FRAMES - 1)))
int16_t audio_buffer[BUFFER_SIZE];
volatile bool isBufferFull = false;
volatile bool isUsingInputTensor = false;
int bufferIndex = 0;

// FFT setup
ArduinoFFT<float> FFT;
float real[FRAME_SIZE];
float imag[FRAME_SIZE];

// TensorFlow Lite Micro globals
tflite::MicroErrorReporter tflErrorReporter;
constexpr int tensorArenaSize = 70 * 1024;  // may need adjustment if AllocateTensors fails
uint8_t tensor_arena[tensorArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model* model = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Quantization params (INT8)
float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;

// Forward declarations
void onPDMdata();
void processBuffer();
void runInference();
void computeMFCC(float* frame, float* mfcc, int frameSize);

// Labels (must match LabelEncoder order used in Python)
const char* labels[NUM_CLASSES] = { "Cow", "Dog", "Frog" };

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Serial.println("Serial initialized.");

  // --------- PDM mic ----------
  PDM.onReceive(onPDMdata);
  PDM.setGain(0);

  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Failed to start PDM!");
    while (1) { ; }
  }
  Serial.println("PDM initialized.");

  // --------- Load TFLite model ----------
  model = tflite::GetModel(animal_sound_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1) { ; }
  }

  static tflite::AllOpsResolver resolver;

  interpreter = new tflite::MicroInterpreter(
    model, resolver, tensor_arena, tensorArenaSize, &tflErrorReporter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed. Try increasing tensorArenaSize.");
    while (1) { ; }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print tensor shapes + types
  Serial.print("Input type: ");
  Serial.println(input->type);
  Serial.print("Output type: ");
  Serial.println(output->type);

  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();

  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();

  // Quantization params
  input_scale = input->params.scale;
  input_zero_point = input->params.zero_point;

  output_scale = output->params.scale;
  output_zero_point = output->params.zero_point;

  Serial.print("Input scale: ");
  Serial.println(input_scale, 10);
  Serial.print("Input zero_point: ");
  Serial.println(input_zero_point);
  Serial.print("Output scale: ");
  Serial.println(output_scale, 10);
  Serial.print("Output zero_point: ");
  Serial.println(output_zero_point);

  // Initialize input tensor to its zero point
    //If any part of your input tensor isn’t overwritten, , you don’t want random memory values.
  memset(input->data.int8, input_zero_point, input->bytes);

  Serial.println("Setup complete.");
}

void loop() {
  if (isBufferFull && !isUsingInputTensor) {
    isUsingInputTensor = true;
    processBuffer();
    isUsingInputTensor = false;
  }
}

void onPDMdata() {
  int bytesAvailable = PDM.available();
  if (bytesAvailable <= 0) return;

  int16_t pdmBuffer[bytesAvailable / 2];
  int samplesRead = PDM.read(pdmBuffer, bytesAvailable) / 2;

  for (int i = 0; i < samplesRead; i++) {
    if (!isBufferFull) {
      audio_buffer[bufferIndex] = pdmBuffer[i];
      bufferIndex++;

      if (bufferIndex >= BUFFER_SIZE) {
        isBufferFull = true;
        bufferIndex = 0;
      }
    }
  }
}

void processBuffer() {
  Serial.println("Processing buffer...");

  // Find max absolute value (for normalization)
  int16_t maxVal = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    int16_t absValue = abs(audio_buffer[i]);
    if (absValue > maxVal) maxVal = absValue;
  }

  // Fill model input: shape expected [1, 13, 200, 1]
  // We flatten as: input[mfcc_index * NUM_FRAMES + frame_index]
  for (int frameIdx = 0; frameIdx < NUM_FRAMES; frameIdx++) {
    float frame[FRAME_SIZE];

    // Copy audio snippet into frame
    for (int j = 0; j < FRAME_SIZE; j++) {
      frame[j] = (float)audio_buffer[frameIdx * HOP_LENGTH + j];
    }

    // Normalize frame to [-1, 1]
    if (maxVal > 0) {
      for (int n = 0; n < FRAME_SIZE; n++) {
        frame[n] = frame[n] / (float)maxVal;
      }
    }

    // Compute MFCC (13 values)
    float mfcc[NUM_MFCC];
    computeMFCC(frame, mfcc, FRAME_SIZE);

    // Quantize + store into int8 input tensor
    for (int k = 0; k < NUM_MFCC; k++) {
      int32_t q = (int32_t)round(mfcc[k] / input_scale + input_zero_point);
      if (q > 127) q = 127;
      if (q < -128) q = -128;

      input->data.int8[k * NUM_FRAMES + frameIdx] = (int8_t)q;
    }
  }

  isBufferFull = false;

  runInference();

  // Reset input tensor to zero point
  memset(input->data.int8, input_zero_point, input->bytes);
}

void runInference() {
  Serial.println("Running inference...");

  TfLiteStatus invokeStatus = interpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  // Read int8 outputs
  int8_t o0 = output->data.int8[0];
  int8_t o1 = output->data.int8[1];
  int8_t o2 = output->data.int8[2];

  // Dequantize (for readable scores)
  float s0 = (o0 - output_zero_point) * output_scale;
  float s1 = (o1 - output_zero_point) * output_scale;
  float s2 = (o2 - output_zero_point) * output_scale;

  // Argmax
  int pred = 0;
  float best = s0;
  if (s1 > best) {
    best = s1;
    pred = 1;
  }
  if (s2 > best) {
    best = s2;
    pred = 2;
  }

  Serial.print("Predicted: ");
  Serial.println(labels[pred]);

  Serial.print("Scores: Cow=");
  Serial.print(s0, 4);
  Serial.print(" Dog=");
  Serial.print(s1, 4);
  Serial.print(" Frog=");
  Serial.println(s2, 4);

  Serial.println();
}

void computeMFCC(float* frame, float* mfcc, int frameSize) {
  // Step 1: Apply Hanning window + copy into FFT buffers
  for (int i = 0; i < frameSize; i++) {
    frame[i] *= 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frameSize - 1)));
    real[i] = frame[i];
    imag[i] = 0.0f;
  }

  // Step 2: FFT
  FFT.compute(real, imag, frameSize, FFT_FORWARD);

  // Step 3: Power spectrum
  float powerSpectrum[FRAME_SIZE / 2];
  for (int i = 0; i < FRAME_SIZE / 2; i++) {
    powerSpectrum[i] = real[i] * real[i] + imag[i] * imag[i];
  }

  // Step 4: Apply Mel filter bank
  float melSpectrum[NUM_MEL_FILTERS] = { 0.0f };
  for (int filter = 0; filter < NUM_MEL_FILTERS; filter++) {
    for (int bin = 0; bin <= FRAME_SIZE / 2; bin++) {
      melSpectrum[filter] += mel_filter_bank[filter][bin] * powerSpectrum[bin];
    }
  }

  // Step 4.5: log scale
  for (int filter = 0; filter < NUM_MEL_FILTERS; filter++) {
    if (melSpectrum[filter] > 0.0f) {
      melSpectrum[filter] = 10.0f * log10f(melSpectrum[filter]);
    }
  }

  // Step 5: DCT
  for (int i = 0; i < NUM_MFCC; i++) {
    mfcc[i] = 0.0f;
    for (int j = 0; j < NUM_MEL_FILTERS; j++) {
      mfcc[i] += melSpectrum[j] * cosf(M_PI * i * (2 * j + 1) / (2 * NUM_MEL_FILTERS));
    }
  }

  // Step 6: Orthonormal normalization
  mfcc[0] *= sqrtf(1.0f / NUM_MEL_FILTERS);
  for (int i = 1; i < NUM_MFCC; i++) {
    mfcc[i] *= sqrtf(2.0f / NUM_MEL_FILTERS);
  }
}