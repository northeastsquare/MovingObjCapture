package facebook.f8demo;

import android.Manifest;
import android.app.ActionBar;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.graphics.YuvImage;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.util.Size;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.TimeUnit;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;
import static java.lang.Math.abs;
import static org.opencv.core.CvType.CV_64FC3;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.cvtColor;


import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.android.Utils;

public class ClassifyCamera extends AppCompatActivity {
    private static final String TAG = "F8DEMO";
    private static final int REQUEST_CAMERA_PERMISSION = 200;

    private TextureView textureView;
    private ImageView bgView, fgView;
    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;
    private TextView tv;
    private String predictedClass = "none";
    private AssetManager mgr;
    private boolean processing = false;
    private Image image = null;
    private boolean run_HWC = false;
    private double[] background = null;
    private boolean hasChanged = false;
    private Mat mYuv = new Mat();
    private Mat mBGR = new Mat();
    private Mat frame = new Mat();
    private Mat fgMask = new Mat();

    private int m_nFrames4BG = 5;
    private BackgroundSubtractor backSub = null;
    private Queue m_qAreaSum;
    private double m_fLastEMA = 0.0;
    double m_nUpEMA = 0;
    double m_nDownEMA = 0;
    private int iImageIdx = -1;
    private double[] tmp = null;

    static {
        System.loadLibrary("native-lib");
        System.loadLibrary("opencv_java3");
        if (!OpenCVLoader.initDebug()) {
            Log.d("opencv", "初始化失败");
        }
    }

    public native String classificationFromCaffe2(int h, int w, byte[] Y, byte[] U, byte[] V,
                                                  int rowStride, int pixelStride, boolean r_hwc);

    public native void initCaffe2(AssetManager mgr);

    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                initCaffe2(mgr);
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        this.requestWindowFeature(Window.FEATURE_NO_TITLE);
        backSub = Video.createBackgroundSubtractorMOG2(5, 16, false);
        m_qAreaSum = new LinkedList();
        mgr = getResources().getAssets();

        new SetUpNeuralNetwork().execute();

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        setContentView(R.layout.activity_classify_camera);

        textureView = (TextureView) findViewById(R.id.textureView);
        textureView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        bgView = (ImageView) findViewById(R.id.bgView);
        bgView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        fgView = (ImageView) findViewById(R.id.fgView);
        fgView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        final GestureDetector gestureDetector = new GestureDetector(this.getApplicationContext(),
                new GestureDetector.SimpleOnGestureListener() {
                    @Override
                    public boolean onDoubleTap(MotionEvent e) {
                        return true;
                    }

                    @Override
                    public void onLongPress(MotionEvent e) {
                        super.onLongPress(e);

                    }

                    @Override
                    public boolean onDoubleTapEvent(MotionEvent e) {
                        return true;
                    }

                    @Override
                    public boolean onDown(MotionEvent e) {
                        return true;
                    }
                });

        textureView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                return gestureDetector.onTouchEvent(event);
            }
        });

        assert textureView != null;
        textureView.setSurfaceTextureListener(textureListener);
        tv = (TextView) findViewById(R.id.sample_text);

    }

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //open your camera here
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };
    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreview();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
        }
    };

    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    protected void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);

            int width = 227;
            int height = 227;
            ImageReader reader = ImageReader.newInstance(width, height, ImageFormat.YUV_420_888, 4);
            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    try {

                        image = reader.acquireNextImage();
                        iImageIdx++;
                        /*if (iImageIdx % 5 != 0)
                        {
                            return;
                        }*/
                        if (processing) {
                            image.close();
                            return;
                        }
                        processing = true;


                        int w = image.getWidth();
                        int h = image.getHeight();
                        ByteBuffer Ybuffer = image.getPlanes()[0].getBuffer();
                        ByteBuffer Ubuffer = image.getPlanes()[1].getBuffer();
                        ByteBuffer Vbuffer = image.getPlanes()[2].getBuffer();

                        int ySize = Ybuffer.remaining();
                        int uSize = Ubuffer.remaining();
                        int vSize = Vbuffer.remaining();

                        byte[] nv21 = new byte[ySize + uSize + vSize];
                        byte[] y = new byte[ySize];
                        byte[] u = new byte[uSize];
                        byte[] v = new byte[vSize];
                        Ybuffer.get(nv21, 0, ySize);
                        Vbuffer.get(nv21, ySize, vSize);
                        Ubuffer.get(nv21, ySize + vSize, uSize);

                        Ybuffer.rewind();
                        Ubuffer.rewind();
                        Vbuffer.rewind();
                        Ybuffer.get(y, 0, ySize);
                        Ubuffer.get(u, 0, uSize);
                        Vbuffer.get(v, 0, vSize);
                        /*for (int i =0;i < ySize;i++){
                            y[i] = (nv21[i]);
                        }
                        int starti = ySize;
                        for (int i =starti;i <starti+ uSize;i++){
                            u[i-starti] = (nv21[i]);
                        }
                        starti = starti+ uSize;
                        for (int i =starti;i <starti+ vSize;i++){
                            v[i-starti] = (nv21[i]);
                        }*/
                        mYuv = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CV_8UC1);
                        mYuv.put(0, 0, nv21);
                        //mBGR = new Mat(image.getHeight(), image.getWidth(), CV_64FC3 );
                        /*tmp = new double[w*h*3];
                        for (int i =0;i < h;i++){
                            for(int j = 0;j<w;j++){
                                double yy = y[i*w+j] & 0xff;
                                double uu = u[i/2*w/2+j/2] & 0xff;
                                double vv = v[i/2*w/2+j/2] & 0xff;
                                //tmp[i*w+j+0]=  yy+1.732446 *(uu-128);//Y + 1.772 (U-V)//black
                                //tmp[i*w+j+1]=yy-(0.698001 * (vv-128)) - (0.337633 * (uu-128));//   0.34414*(uu-128)-0.71414*(vv-128);//G = Y - 0.34414 (U-128) - 0.71414 (V-128)
                                //tmp[i*w+j+2]=yy+(1.370705 * (vv-128));//1.402*(vv-128);//R = Y + 1.402 (V-128)
                                tmp[i*w*3+3*j+0] = yy + 1.770*( uu-128);//128.0);
                                tmp[i*w*3+3*j+1] = yy-0.343*(uu-128.0)-0.714*(vv-128.0);
                                tmp[i*w*3+3*j+2] = yy+1.403*(vv-128.0);
                            }
                        }
                        mBGR = new Mat(h, w, CV_64FC3);
                        mBGR.put(0, 0,tmp);*/

                        //mYuv = imageToMat( image);
                        Mat mGray = new Mat();
                        cvtColor(mYuv, mGray, Imgproc.COLOR_YUV2GRAY_NV21, 3);
                        cvtColor(mYuv, mBGR, Imgproc.COLOR_YUV420sp2BGR, 3);
                        frame = mGray;
                        backSub.apply(mBGR, fgMask);

                        //! [display_frame_number]
                        // get the frame number and write it on the current frame
                        Imgproc.rectangle(frame, new Point(10, 2), new Point(100, 20), new Scalar(255, 255, 255), -1);



                        /*String frameNumberString = String.format("%d", (int)capture.get(Videoio.CAP_PROP_POS_FRAMES));
                        Imgproc.putText(frame, frameNumberString, new Point(15, 15), Core.FONT_HERSHEY_SIMPLEX, 0.5,
                                new Scalar(0, 0, 0));*/
                        //! [display_frame_number]


                        // TODO: use these for proper image processing on different formats.
                        int rowStride = image.getPlanes()[1].getRowStride();
                        int pixelStride = image.getPlanes()[1].getPixelStride();
                        Ybuffer.rewind();
                        Ubuffer.rewind();
                        Vbuffer.rewind();
                        byte[] Y = new byte[Ybuffer.capacity()];
                        byte[] U = new byte[Ubuffer.capacity()];
                        byte[] V = new byte[Vbuffer.capacity()];
                        Ybuffer.get(Y);
                        Ubuffer.get(U);
                        Vbuffer.get(V);
                        hasChanged = false;
                        double rate = 0.0;
                        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                        Mat hierarchy = new Mat();
                        Imgproc.findContours(fgMask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

                        Iterator<MatOfPoint> iterator = contours.iterator();
                        double area_sum = 0;
                        while (iterator.hasNext()) {
                            MatOfPoint contour = iterator.next();
                            double area = Imgproc.contourArea(contour);
                            Log.i(TAG, "area:" + area + " area_sum:" + area_sum);
                            if (area > 100) {
                                area_sum += area;
                            }
                        }
                        //double sma = 0.0;
                        double ema = 0.0;

                        if (m_qAreaSum.size() < m_nFrames4BG) {
                            m_qAreaSum.add(area_sum);
                        } else {
                            m_qAreaSum.remove();
                            m_qAreaSum.add(area_sum);
                        }
                        Iterator it = m_qAreaSum.iterator();
                        double s = 0;
                        for (Object as : m_qAreaSum) {
                            s += (double) (as);
                        }
                        sma = s / m_qAreaSum.size();
                        ema = (area_sum - m_fLastEMA) * 2 / (1 + m_qAreaSum.size()) + m_fLastEMA;
                        if (ema >= m_fLastEMA) {
                            m_nUpEMA++;
                            m_nDownEMA = 0;
                        } else {
                            m_nUpEMA = 0;
                            m_nDownEMA++;
                        }
                        if (m_nDownEMA < 3) {
                            return;
                        }

                        m_fLastEMA = ema;

//                        if (area_sum<1000){
////                            Toast.makeText(ClassifyCamera.this,
////                                    "area_sum:"+area_sum,
////                                    Toast.LENGTH_SHORT).show();
////                            processing = false;
////                          return;
////                          }
                        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED))
                        // 判断是否可以对SDcard进行操作
                        {      // 获取SDCard指定目录下
                            Date c = Calendar.getInstance().getTime();
                            Log.i(TAG, "Current time => " + c);
                            SimpleDateFormat datef = new SimpleDateFormat("yyyy-MM-dd");
                            SimpleDateFormat timef = new SimpleDateFormat("HH-mm-ss");

                            String formattedDate = datef.format(c);
                            String formattedTime = timef.format(c);
                            String sdCardDir = Environment.getExternalStorageDirectory() + "/CoolImage/" + formattedDate + "/";
                            File dirFile = new File(sdCardDir);  //目录转化成文件夹
                            if (!dirFile.exists()) {                //如果不存在，那就建立这个文件夹
                                //dirFile .mkdirs();
                                Toast.makeText(ClassifyCamera.this,
                                        (dirFile.mkdirs() ? "Directory has been created" : "Directory not created"),
                                        Toast.LENGTH_SHORT).show();
                            }
                            //文件夹有啦，就可以保存图片啦
                            String fname = formattedTime + "-" + System.currentTimeMillis() + ".jpg";
                            String fullname = sdCardDir + fname;
                            Imgcodecs.imwrite(fullname, mBGR);
                            Log.i(TAG, "保存到_sd_指定目录文件夹下_" + fullname);
                        }
                        predictedClass = classificationFromCaffe2(h, w, Y, U, V,
                                rowStride, pixelStride, run_HWC);
                        predictedClass = predictedClass;  //+ " rate:"+rate;
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                tv.setText(predictedClass);
                                //String ss = "tview";
                                //tv.setText(ss);
                                //Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2RGBA, 4);
                                Bitmap bgbmp = Bitmap.createBitmap(frame.width(), frame.height(), Bitmap.Config.ARGB_8888);
                                Utils.matToBitmap(frame, bgbmp);
                                bgView.setImageBitmap(bgbmp);
                                bgView.invalidate();
                                bgView.setVisibility(View.VISIBLE);

                                //Imgproc.cvtColor(fgMask, fgMask, Imgproc.COLOR_GRAY2RGBA, 4);
                                Bitmap fgbmp = Bitmap.createBitmap(frame.width(), frame.height(), Bitmap.Config.ARGB_8888);
                                Utils.matToBitmap(fgMask, fgbmp);
                                fgbmp = Bitmap.createScaledBitmap(fgbmp, fgView.getWidth(), fgView.getHeight(), true);
                                fgView.setImageBitmap(fgbmp);
                                fgView.invalidate();
                                fgView.setVisibility(View.VISIBLE);

                            }


                        });
                        processing = false;


                    } finally {
                        if (image != null) {
                            image.close();
                        }
                    }
                }
            };
            reader.setOnImageAvailableListener(readerListener, mBackgroundHandler);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);
            captureRequestBuilder.addTarget(reader.getSurface());

            cameraDevice.createCaptureSession(Arrays.asList(surface, reader.getSurface()), new CameraCaptureSession.StateCallback() {
                //cameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCaptureSession.StateCallback(){
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    if (null == cameraDevice) {
                        return;
                    }
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(ClassifyCamera.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(ClassifyCamera.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    protected void updatePreview() {
        if (null == cameraDevice) {
            return;
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults[0] == PackageManager.PERMISSION_DENIED) {
                Toast.makeText(ClassifyCamera.this, "You can't use this app without granting permission", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startBackgroundThread();
        if (textureView.isAvailable()) {
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(textureListener);
        }
    }

    @Override
    protected void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }
}
