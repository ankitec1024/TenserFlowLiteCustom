/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package me.ndres.tflitedemo;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.text.SpannableString;
import android.text.SpannableStringBuilder;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Switch;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.NetworkResponse;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.VolleyLog;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.Interpreter;

import static android.content.Context.MODE_PRIVATE;
import static me.ndres.tflitedemo.Camera2BasicFragment.test;

/**
 * Classifies images with Tensorflow Lite.
 */
public abstract class ImageClassifier {
  // Display preferences
  private static final float GOOD_PROB_THRESHOLD = 0.3f;
  private static final int SMALL_COLOR = 0xffddaa88;

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteDemo";

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 1;

  /** Preallocated buffers for storing image data in. */
  private int[] intValues = new int[getImageSizeX() * getImageSizeY()];

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModel;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  protected ByteBuffer imgData = null;

  /** multi-stage low pass filter * */
  private float[][] filterLabelProbArray = null;

  private static final int FILTER_STAGES = 3;
  private static final float FILTER_FACTOR = 0.4f;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });

  private PriorityQueue<Map.Entry<String, Float>> kkkk =
          new PriorityQueue<>(
                  RESULTS_TO_SHOW,
                  new Comparator<Map.Entry<String, Float>>() {
                    @Override
                    public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                      return (o1.getValue()).compareTo(o2.getValue());
                    }
                  });


  /** holds a gpu delegate */
  Delegate gpuDelegate = null;

  /** Initializes an {@code ImageClassifier}. */
  ImageClassifier(Activity activity) throws IOException {
    tfliteModel = loadModelFile(activity);
    tflite = new Interpreter(tfliteModel, tfliteOptions);
    labelList = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE
                * getImageSizeX()
                * getImageSizeY()
                * DIM_PIXEL_SIZE
                * getNumBytesPerChannel());
    imgData.order(ByteOrder.nativeOrder());
    filterLabelProbArray = new float[FILTER_STAGES][getNumLabels()];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }

  /** Classifies a frame from the preview stream. */
  void classifyFrame(Bitmap bitmap, SpannableStringBuilder builder,  Activity activity) {
    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      builder.append(new SpannableString("Uninitialized Classifier."));
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    runInference();
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // Smooth the results across frames.
    applyFilter();

    // Print the results.
    printTopKLabels(builder, activity);
    long duration = endTime - startTime;
//    SpannableString span = new SpannableString(duration + " ms"); //an comment..................when face comes in the front 4
//    span.setSpan(new ForegroundColorSpan(android.graphics.Color.LTGRAY), 0, span.length(), 0);
//    builder.append(span);
  }

  void applyFilter() {
    int numLabels = getNumLabels();

    // Low pass filter `labelProbArray` into the first stage of the filter.
    for (int j = 0; j < numLabels; ++j) {
      filterLabelProbArray[0][j] +=
          FILTER_FACTOR * (getProbability(j) - filterLabelProbArray[0][j]);
    }
    // Low pass filter each stage into the next.
    for (int i = 1; i < FILTER_STAGES; ++i) {
      for (int j = 0; j < numLabels; ++j) {
        filterLabelProbArray[i][j] +=
            FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
      }
    }

    // Copy the last stage filter output back to `labelProbArray`.
    for (int j = 0; j < numLabels; ++j) {
      setProbability(j, filterLabelProbArray[FILTER_STAGES - 1][j]);
    }
  }

  private void recreateInterpreter() {
    if (tflite != null) {
      tflite.close();
      // TODO(b/120679982)
      // gpuDelegate.close();
      tflite = new Interpreter(tfliteModel, tfliteOptions);
    }
  }

  public void useGpu() {
    if (gpuDelegate == null && GpuDelegateHelper.isGpuDelegateAvailable()) {
      gpuDelegate = GpuDelegateHelper.createGpuDelegate();
      tfliteOptions.addDelegate(gpuDelegate);
      recreateInterpreter();
    }
  }

  public void useCPU() {
    tfliteOptions.setUseNNAPI(false);
    recreateInterpreter();
  }

  public void useNNAPI() {
    tfliteOptions.setUseNNAPI(true);
    recreateInterpreter();
  }

  public void setNumThreads(int numThreads) {
    tfliteOptions.setNumThreads(numThreads);
    recreateInterpreter();
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
    tfliteModel = null;
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath()))); //an comment.................getLabelPath() --all label text which is shown on ui thread(sad, happy, angry etc.)
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < getImageSizeX(); ++i) {
      for (int j = 0; j < getImageSizeY(); ++j) {
        final int val = intValues[pixel++];
        addPixelValue(val);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private void printTopKLabels(SpannableStringBuilder builder, Activity activity) {//an commentttttttttttttttttttt.............................printing the value of the gesture
    for (int i = 0; i < getNumLabels(); ++i) {
      sortedLabels.add(
          new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));//an comment..............A PriorityQueue is used for sorting. Each Classifier subclass has a getNormalizedProbability method, which is expected to return a probability between 0 and 1 of a given class being represented by the image.
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }

      kkkk.add(new AbstractMap.SimpleEntry<>(labelList.get(i), getNormalizedProbability(i)));
    }

    final int size = sortedLabels.size();
    int arr[] = new int[10];


//    //an commenttttttttttttttttt.................................................

    if(System.currentTimeMillis() - test > 5000 ){

      test = System.currentTimeMillis();


      if(Camera2BasicFragment.bitmapToBeUpload != null)
        callssLinkApi(Camera2BasicFragment.bitmapToBeUpload, activity);

//      Map.Entry<String, Float> label = (Map.Entry<String, Float>) sortedLabels;
//      String temp = label.getKey();


    }

      for(int l=0; l<size; l++){
          Map.Entry<String, Float> label = sortedLabels.poll();
          if(l == size -1){
              String temp = label.getKey();
              int temp1 = (int) (label.getValue() * 100);
              drawGesture(temp, activity, temp1);
          }
      }
//
//    //an commenttttttttttttttttt.................................................

//    for (int i = 0; i < size; i++) {//
//      Map.Entry<String, Float> label = sortedLabels.poll();//
//      SpannableString span =
//          new SpannableString(String.format("%s: %4.2f\n", label.getKey(), label.getValue()));//
//      int color;//
//      // Make it white when probability larger than threshold.
//      if (label.getValue() > GOOD_PROB_THRESHOLD) {//
//        color = android.graphics.Color.WHITE;//
//      } else {
//        color = SMALL_COLOR;//
//      }
//      // Make first item bigger.
//      if (i == size - 1) {//
//        float sizeScale = (i == size - 1) ? 1.25f : 0.8f;//
//        span.setSpan(new RelativeSizeSpan(sizeScale), 0, span.length(), 0);//
//      }
//      span.setSpan(new ForegroundColorSpan(color), 0, span.length(), 0);
//      builder.insert(0, span);
//    }
  }


  private void callFiveSecondsApi(Activity activity) {


    try {
      RequestQueue requestQueue = Volley.newRequestQueue(activity);
      final String mRequestBody = makeJsonObjectParams(activity);

      StringRequest stringRequest = new StringRequest(Request.Method.POST, "https://mvp.verify24x7.in/emotional/api/tasks/saveEmotions", new Response.Listener<String>() {
        @Override
        public void onResponse(String response) {
          try {
            if (new JSONObject(response).getInt("statusCode") == 200) {



            }
          } catch (JSONException e) {
            e.printStackTrace();
          }

        }
      }, new Response.ErrorListener() {
        @Override
        public void onErrorResponse(VolleyError error) {
          Toast.makeText(activity, "errror", Toast.LENGTH_LONG).show();
        }
      }) {
        @Override
        public String getBodyContentType() {
          return "application/json; charset=utf-8";
        }


        @Override
        public byte[] getBody() throws AuthFailureError {
          try {
            return mRequestBody == null ? null : mRequestBody.getBytes("utf-8");
          } catch (UnsupportedEncodingException uee) {
            VolleyLog.wtf("Unsupported Encoding while trying to get the bytes of %s using %s", mRequestBody, "utf-8");
            return null;
          }
        }

        @Override
        protected Response<String> parseNetworkResponse(NetworkResponse response) {
          String responseString = "";
          if (response != null) {

            responseString = String.valueOf(response.statusCode);

          }
//                    return Response.success(responseString, HttpHeaderParser.parseCacheHeaders(response));
          return super.parseNetworkResponse(response);
        }
      };
      requestQueue = Volley.newRequestQueue(activity);

      requestQueue.add(stringRequest);
    } catch (Exception e) {//JSONException e
      e.printStackTrace();
    }


  }

  String temper1="", temper2="", temper3="", temper4="", temper5="", temper6="", temper7="";
  ArrayList<Float> temper11 = new ArrayList();
  ArrayList<Float> temper22 = new ArrayList();
  ArrayList<Float> temper33 = new ArrayList();
  ArrayList<Float> temper44 = new ArrayList();
  ArrayList<Float> temper55 = new ArrayList();
  ArrayList<Float> temper66 = new ArrayList();
  ArrayList<Float> temper77 = new ArrayList();

  private String makeJsonObjectParams(Activity activity) {

    final SharedPreferences sharedPreferences = activity.getSharedPreferences("NIGHT_MODE", MODE_PRIVATE);

    for(int l=0; l < kkkk.size(); l++){

        Map.Entry<String, Float> label = kkkk.poll();

      if(label.getKey().equalsIgnoreCase("angry")){
        temper11.add(label.getValue());
      }
      else if(label.getKey().equalsIgnoreCase("disgust")){
        temper22.add(label.getValue());
      }
      else if(label.getKey().equalsIgnoreCase("scared")){
        temper33.add(label.getValue());
      }
      else if(label.getKey().equalsIgnoreCase("happy")){
        temper44.add(label.getValue());
      }
      else if(label.getKey().equalsIgnoreCase("sad")){
        temper55.add(label.getValue());
      }
      else if(label.getKey().equalsIgnoreCase("surprised")){
        temper66.add(label.getValue());
      }
      else if(label.getKey().equalsIgnoreCase("neutral")){
        temper77.add(label.getValue());
      }

    }

    JSONArray mediaData = new JSONArray();
    try {

      JSONObject temp = new JSONObject();
      temp.put("angry", temper11);

      mediaData.put(temp);

      //..............................................................

      JSONObject temp1 = new JSONObject();
      temp1.put("disgust", temper22);

      mediaData.put(temp1);

      //..............................................................

      JSONObject temp2 = new JSONObject();
      temp2.put("scared", temper33);

      mediaData.put(temp2);

      //..............................................................

      JSONObject temp3 = new JSONObject();
      temp3.put("happy", temper44);

      mediaData.put(temp3);

      //..............................................................

      JSONObject temp4 = new JSONObject();
      temp4.put("sad", temper55);

      mediaData.put(temp4);

      //..............................................................

      JSONObject temp5 = new JSONObject();
      temp5.put("surprised", temper66);

      mediaData.put(temp5);

      //..............................................................

      JSONObject temp6 = new JSONObject();
      temp6.put("neutral", temper77);

      mediaData.put(temp6);

      //..............................................................


    } catch (ArrayIndexOutOfBoundsException e) {
      e.printStackTrace();
    } catch (JSONException e) {
      e.printStackTrace();
    }


    JSONObject data1 = new JSONObject();
    try {


      data1.put("username", sharedPreferences.getString("name",""));
      data1.put("photo", sslink);
      data1.put("emotions", mediaData);

    } catch (JSONException e) {
      e.printStackTrace();
    }


    JSONObject data = new JSONObject();
    try {
    data.put("data", data1);
    } catch (JSONException e) {
      e.printStackTrace();
    }

    final String mRequestBody = data.toString();

    return mRequestBody;
  }

  private void drawGesture(String gesture, Activity activity, int percentage) {


      activity.runOnUiThread(new Runnable() {

        @Override
        public void run() {

          if(Camera2BasicFragment.imageEmoji.getVisibility() != View.VISIBLE ){
            Camera2BasicFragment.imageEmoji.setVisibility(View.VISIBLE);
          }

          switch (gesture){
            case "angry":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tenserangry);
              Camera2BasicFragment.textEmoji.setText("Angry "+percentage+"%");
              break;
            case "disgust":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tenserdisgust);
              Camera2BasicFragment.textEmoji.setText("Disgust "+percentage+"%");
              break;
            case "scared":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tenserscared);
              Camera2BasicFragment.textEmoji.setText("Scared "+percentage+"%");
              break;
            case "happy":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tensersmile);
              Camera2BasicFragment.textEmoji.setText("Happy "+percentage+"%");
              break;
            case "sad":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tensersad);
              Camera2BasicFragment.textEmoji.setText("Sad "+percentage+"%");
              break;
            case "surprised":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tensersurprised);
              Camera2BasicFragment.textEmoji.setText("Surprised "+percentage+"%");
              break;
            case "neutral":
              Camera2BasicFragment.imageEmoji.setImageResource(R.drawable.tenserneutral);
              Camera2BasicFragment.textEmoji.setText("Neutral "+percentage+"%");
              break;
          }

        }
      });

    }








  public static RequestQueue rQueue;
  private String sslink;
  private void callssLinkApi(Bitmap bitmap, Activity activity) {


    VolleyMultipartRequest volleyMultipartRequest = new VolleyMultipartRequest(Request.Method.POST, "https://mvp.verify24x7.in/adv/add/task/uploadImage?username=" + "Ankit",
            new Response.Listener<NetworkResponse>() {
              @Override
              public void onResponse(NetworkResponse response) {

                Log.d("ressssssoo", new String(response.data));
                try {
                  JSONObject jo = new JSONObject(new String(response.data));
                  if (jo.getInt("statusCode") == 200) {


                    JSONObject jsonObject = new JSONObject(new String(response.data));

//                    download_url.add(jsonObject.getJSONObject("dataObj").getString("imageLink"));
                    sslink = jsonObject.getJSONObject("dataObj").getString("imageLink");

                    callFiveSecondsApi(activity);


                  } else {

                  }

                } catch (JSONException e) {
                  e.printStackTrace();
                }


              }
            },
            new Response.ErrorListener() {
              @Override
              public void onErrorResponse(VolleyError error) {
                Toast.makeText(activity, ".....", Toast.LENGTH_SHORT).show();
              }
            }) {


      @Override
      protected Map<String, String> getParams() throws AuthFailureError {
        Map<String, String> params = new HashMap<>();
        return params;
      }

      /*
       *pass files using below method
       * */
      @Override
      public Map<String, DataPart> getByteData() {
        Map<String, DataPart> params = new HashMap<>();
        long imagename = System.currentTimeMillis();

        params.put("image", new DataPart(imagename + ".png", getFileDataFromDrawable(bitmap)));

        return params;
      }
    };


    volleyMultipartRequest.setRetryPolicy(new DefaultRetryPolicy(
            0,
            DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
            DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
    rQueue = Volley.newRequestQueue(activity);
    rQueue.add(volleyMultipartRequest);

  }

  public byte[] getFileDataFromDrawable(Bitmap bitmap) {
    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.PNG, 0, byteArrayOutputStream);
    return byteArrayOutputStream.toByteArray();
  }











    /**
   * Get the name of the model file stored in Assets.
   *
   * @return
   */
  protected abstract String getModelPath();

  /**
   * Get the name of the label file stored in Assets.
   *
   * @return
   */
  protected abstract String getLabelPath();//an comment.................

  /**
   * Get the image size along the x axis.
   *
   * @return
   */
  protected abstract int getImageSizeX();

  /**
   * Get the image size along the y axis.
   *
   * @return
   */
  protected abstract int getImageSizeY();

  /**
   * Get the number of bytes that is used to store a single color channel value.
   *
   * @return
   */
  protected abstract int getNumBytesPerChannel();

  /**
   * Add pixelValue to byteBuffer.
   *
   * @param pixelValue
   */
  protected abstract void addPixelValue(int pixelValue);

  /**
   * Read the probability value for the specified label This is either the original value as it was
   * read from the net's output or the updated value after the filter was applied.
   *
   * @param labelIndex
   * @return
   */
  protected abstract float getProbability(int labelIndex);

  /**
   * Set the probability value for the specified label.
   *
   * @param labelIndex
   * @param value
   */
  protected abstract void setProbability(int labelIndex, Number value);

  /**
   * Get the normalized probability value for the specified label. This is the final value as it
   * will be shown to the user.
   *
   * @return
   */
  protected abstract float getNormalizedProbability(int labelIndex);

  /**
   * Run inference using the prepared input in {@link #imgData}. Afterwards, the result will be
   * provided by getProbability().
   *
   * <p>This additional method is necessary, because we don't have a common base for different
   * primitive data types.
   */
  protected abstract void runInference();

  /**
   * Get the total number of labels.
   *
   * @return
   */
  protected int getNumLabels() {
    return labelList.size();
  }
}
