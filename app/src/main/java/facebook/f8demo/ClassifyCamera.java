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
import android.view.ViewTreeObserver;
import android.view.Window;

import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
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
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;
import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static org.opencv.core.Core.sumElems;
import static org.opencv.core.CvType.CV_64FC3;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.moments;


import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.video.BackgroundSubtractor;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.android.Utils;
import facebook.f8demo.StorageHelper;
import facebook.f8demo.TimeRunnable;

public class ClassifyCamera extends AppCompatActivity {
    private static final String TAG = "F8DEMO";
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private TextureView textureView;
    private ViewTreeObserver m_cViewTreeObserver = null;
    int m_iCameraOrientation = 0;
    int m_iTextureViewHeight=1;
    int m_iTextureViewWidth=1;
    private DrawRectangle m_cDrawRectangle;
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
    Bitmap bgbmp;
    Bitmap fgbmp;
    Mat imResized = new Mat();
    private  Date m_dtLastSaveDate;
    private  long m_lLastMilliSeconds=0;
    long t4 = 0,t0=0;
    int[] m_aPoint = new int[8];
    int m_iArrayIndex = 0;

    private int m_nFrames4BG = 5;
    private BackgroundSubtractor backSub = null;
    private Queue m_qAreaSum;
    private Queue m_qContourCenter;
    private double m_fLastEMA = 0.0;
    //private double m_fLastContour2Center = 0.0;
    double m_nUpEMA = 0;
    double m_nDownEMA = 0;
    double m_nAwayFrame=0;
    private int iImageIdx = -1;
    private double[] tmp = null;
    private StorageHelper m_cStorageHelper= null;
    private String m_sExternalRoot = null;
    String fullname;
    ArrayList<MatOfPoint> m_alScale = new ArrayList<MatOfPoint>();

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
        m_qContourCenter = new LinkedList();
        mgr = getResources().getAssets();
//        m_cStorageHelper = new StorageHelper();
//        List<StorageHelper.StorageVolume> lst= m_cStorageHelper.getStorages(true);
//        for (StorageHelper.StorageVolume sv:lst){
//            if(sv.isRemovable() == true){
//                m_sExternalRoot = sv.file.toString();
//            }
//        }
        new SetUpNeuralNetwork().execute();

        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        setContentView(R.layout.activity_classify_camera);

        textureView = (TextureView) findViewById(R.id.textureView);
        textureView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        m_cViewTreeObserver = textureView.getViewTreeObserver();
        m_cViewTreeObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
            @Override
            public void onGlobalLayout() {
                // TODO Auto-generated method stub
                m_iTextureViewHeight = textureView.getHeight();
                m_iTextureViewWidth = textureView.getWidth();
                textureView.getViewTreeObserver().removeGlobalOnLayoutListener(
                        this);
                m_aPoint[0]=0;
                m_aPoint[1]=0;
                m_aPoint[2]=m_iTextureViewWidth-1;
                m_aPoint[3]=0;
                m_aPoint[4]=m_iTextureViewWidth-1;
                m_aPoint[5]=m_iTextureViewHeight-1;
                m_aPoint[6]=0;
                m_aPoint[7]=m_iTextureViewHeight-1;
                Toast.makeText(ClassifyCamera.this,"wh:"+m_iTextureViewWidth+" "+m_iTextureViewHeight,3000).show();
                Log.i(TAG, "m_iTextureViewHeight:"+m_iTextureViewHeight+"  "+m_iTextureViewWidth);
            }
        });



        LinearLayout surfaceView =  (LinearLayout)findViewById(R.id.surface);
        m_cDrawRectangle = new DrawRectangle(this);
        surfaceView.addView(m_cDrawRectangle);

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
                        int x = (int) e.getX(0);
                        int y = (int) e.getY(0);
                        Log.i(TAG, "x y:"+x+" "+y);
                        m_aPoint[m_iArrayIndex++] = x;
                        m_aPoint[m_iArrayIndex++] = y;
                        if (m_iArrayIndex==8){
                            m_cDrawRectangle.setPTS(m_aPoint);
                        }
                        m_iArrayIndex = m_iArrayIndex % 8;

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
    public static Mat imageToMatSimple(Image image){
        ByteBuffer Ybuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer Ubuffer = image.getPlanes()[1].getBuffer();
        ByteBuffer Vbuffer = image.getPlanes()[2].getBuffer();

        int ySize = Ybuffer.remaining();
        int uSize = Ubuffer.remaining();
        int vSize = Vbuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        Ybuffer.get(nv21, 0, ySize);
//        Ubuffer.get(nv21, ySize, uSize);
//        Vbuffer.get(nv21, ySize + uSize, vSize);
        Vbuffer.get(nv21, ySize, vSize);
        Ubuffer.get(nv21, ySize + vSize, uSize);
        Mat mat = new Mat(image.getHeight() + image.getHeight() / 2, image.getWidth(), CV_8UC1);
        mat.put(0, 0, nv21);
        return mat;
    }
    public static Mat imageToMat(Image image) {
        ByteBuffer buffer;
        int rowStride;
        int pixelStride;
        int width = image.getWidth();
        int height = image.getHeight();
        int offset = 0;

        Image.Plane[] planes = image.getPlanes();
        byte[] data = new byte[image.getWidth() * image.getHeight() * ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8];
        byte[] rowData = new byte[planes[0].getRowStride()];

        for (int i = 0; i < planes.length; i++) {
            buffer = planes[i].getBuffer();
            rowStride = planes[i].getRowStride();
            pixelStride = planes[i].getPixelStride();
            int w = (i == 0) ? width : width / 2;
            int h = (i == 0) ? height : height / 2;
            for (int row = 0; row < h; row++) {
                int bytesPerPixel = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888) / 8;
                if (pixelStride == bytesPerPixel) {
                    int length = w * bytesPerPixel;
                    buffer.get(data, offset, length);

                    if (h - row != 1) {
                        buffer.position(buffer.position() + rowStride - length);
                    }
                    offset += length;
                } else {


                    if (h - row == 1) {
                        buffer.get(rowData, 0, width - pixelStride + 1);
                    } else {
                        buffer.get(rowData, 0, rowStride);
                    }

                    for (int col = 0; col < w; col++) {
                        data[offset++] = rowData[col * pixelStride];
                    }
                }
            }
        }

        Mat mat = new Mat(height + height / 2, width, CV_8UC1);
        mat.put(0, 0, data);

        return mat;
    }
    public static Mat imageToMatFaster(Image image) {

        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width*height;
        int uvSize = width*height/4;

        byte[] nv21 = new byte[ySize + uvSize*2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStride = image.getPlanes()[0].getRowStride();
        assert(image.getPlanes()[0].getPixelStride() == 1);

        int pos = 0;

        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize);
            pos += ySize;
        }
        else {
            long yBufferPos = width - rowStride; // not an actual position
            for (; pos<ySize; pos+=width) {
                yBufferPos += rowStride - width;
                yBuffer.position((int)yBufferPos);
                yBuffer.get(nv21, pos, width);
            }
        }

        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        assert(rowStride == image.getPlanes()[1].getRowStride());
        assert(pixelStride == image.getPlanes()[1].getPixelStride());

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            // maybe V an U planes overlap as per NV21, which means vBuffer[1] is alias of uBuffer[0]
            byte savePixel = vBuffer.get(1);
            vBuffer.put(1, (byte)0);
            if (uBuffer.get(0) == 0) {
                vBuffer.put(1, (byte)255);
                if (uBuffer.get(0) == 255) {
                    vBuffer.put(1, savePixel);
                    vBuffer.get(nv21, ySize, uvSize);

                    //return nv21; // shortcut
                    Mat mat = new Mat(height + height / 2, width, CV_8UC1);
                    mat.put(0, 0, nv21);
                    return mat;
                }
            }

            // unfortunately, the check failed. We must save U and V pixel by pixel
            vBuffer.put(1, savePixel);
        }

        // other optimizations could check if (pixelStride == 1) or (pixelStride == 2),
        // but performance gain would be less significant

        for (int row=0; row<height/2; row++) {
            for (int col=0; col<width/2; col++) {
                int vuPos = col*pixelStride + row*rowStride;
                nv21[pos++] = vBuffer.get(vuPos);
                nv21[pos++] = uBuffer.get(vuPos);
            }
        }

        //return nv21;
        Mat mat = new Mat(height + height / 2, width, CV_8UC1);
        mat.put(0, 0, nv21);

        return mat;
    }


    protected void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);
            int screenWidth = this.getApplicationContext().getResources().getDisplayMetrics().widthPixels;
            int screenHeight = this.getApplicationContext().getResources().getDisplayMetrics().heightPixels;

//            m_iTextureViewHeight = textureView.getHeight();
//            m_iTextureViewWidth = textureView.getWidth();

            ImageReader reader = ImageReader.newInstance(screenWidth, screenHeight, ImageFormat.YUV_420_888, 4);
            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    try {
                        //long
                        //t0 = System.currentTimeMillis();
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


                        //long t1 = System.currentTimeMillis();
                        //Log.i(TAG, "timing t01:"+(t1-t0));
                        mYuv = imageToMat(image);
                        //mYuv = imageToMatFaster(image);

                        //long t2 = System.currentTimeMillis();
                       // Log.i(TAG, "timing t12:"+(t2-t1));
                        //mYuv = imageToMatSimple(image);

                        //Mat mGray = new Mat();
                        //Mat mRGB = new Mat();
                        //cvtColor(mYuv, mGray, Imgproc.COLOR_YUV2GRAY_NV21, 3);
                        cvtColor(mYuv, mBGR, Imgproc.COLOR_YUV2BGR_I420, 3);
                        //cvtColor(mYuv, mRGB, Imgproc.COLOR_YUV2RGBA_I420, 4);
                        //frame = mRGB;
                        int width = mBGR.width();// image.getWidth();
                        int height = mBGR.height();//image.getHeight();
                        Log.i(TAG, "before roate height width:"+width+" "+height);
                        if (m_iCameraOrientation == 270) {
                            // Rotate clockwise 270 degrees
                            //Core.flip(mBGR.t(), mBGR, 0);
                            Core.rotate(mBGR, mBGR, Core.ROTATE_90_COUNTERCLOCKWISE);
                        } else if (m_iCameraOrientation == 180) {
                            // Rotate clockwise 180 degrees
                            //Core.flip(mBGR, mBGR, -1);
                            Core.rotate(mBGR, mBGR, Core.ROTATE_180);
                        } else if (m_iCameraOrientation == 90) {
                            //  90 dRotate clockwiseegrees
                            //Core.flip(mBGR.t(), mBGR, 1);
                            Core.rotate(mBGR, mBGR, Core.ROTATE_90_CLOCKWISE);
                        }
                        width = mBGR.width();// image.getWidth();
                        height = mBGR.height();//image.getHeight();
                        //long t3 = System.currentTimeMillis();
                        //Log.i(TAG, "timing t23:"+(t3-t2));
                        double rate = Math.sqrt((300.*400)/(width*height));
                        int neww = (int)(width*rate);
                        int newh = (int)(height*rate);
                        Log.i(TAG, "neww newh1:"+neww + "    "+newh+"  height width:"+width+" "+height);
                        Imgproc.resize(mBGR,imResized, new org.opencv.core.Size(neww, newh));
                        neww = imResized.width();
                        newh = imResized.height();
                        Log.i(TAG, "neww newh2:"+neww + "    "+newh);
                        bgbmp = Bitmap.createBitmap(imResized.width(), imResized.height(), Bitmap.Config.ARGB_8888);

                        //get quadrilateral mask
                        double xrate = 1.0*neww/m_iTextureViewWidth;
                        double yrate = 1.0*newh/m_iTextureViewHeight;
                        int [] impts = new int[8];
                        impts[0] = (int)(m_aPoint[0]*xrate);
                        impts[1] = (int)(m_aPoint[1]*yrate);
                        impts[2] = (int)(m_aPoint[2]*xrate);
                        impts[3] = (int)(m_aPoint[3]*yrate);
                        impts[4] = (int)(m_aPoint[4]*xrate);
                        impts[5] = (int)(m_aPoint[5]*yrate);
                        impts[6] = (int)(m_aPoint[6]*xrate);
                        impts[7] = (int)(m_aPoint[7]*yrate);

                        List<MatOfPoint> border = new ArrayList<MatOfPoint>();
                        border.add(new MatOfPoint(new Point(impts[0],impts[1]), new Point(impts[2],impts[3]), new Point(impts[4],impts[5]), new Point(impts[6],impts[7])));
                        Mat mask = Mat.zeros(newh, neww, CV_8UC3);
                        Imgproc.fillPoly(mask, border, new Scalar(255,255,255));
                        Core.bitwise_and(imResized, mask, imResized);
                        Core.divide(mask, new Scalar(255,255,255), mask);//before i use (255,255,255),then Core.sumElems(mask).val:1,0,0
                        double[] sum_each_channel = Core.sumElems(mask).val;
                        int nMaskPixelNum = (int) (sum_each_channel[0]+sum_each_channel[1]+sum_each_channel[2])/3;
                        //int nMaskPixelNum = (int)(Core.sumElems()/3);//here I spent much time on
                        Log.i("TAG", "nMaskPixelNum:"+nMaskPixelNum+ " sum:"+neww*newh+" mask channels:"+mask.channels());
                        Log.i(TAG, " maxVal:"+ Core.minMaxLoc(mask.reshape(1, (int)mask.total())).maxVal);

                        Mat mrgba = new Mat();
                        cvtColor(imResized, mrgba, Imgproc.COLOR_BGR2RGBA, 4);
                        Utils.matToBitmap(mrgba, bgbmp);
                        backSub.apply(imResized, fgMask);
                        Mat dkernel = Imgproc.getStructuringElement(Imgproc.MORPH_DILATE, new org.opencv.core.Size(5, 5), new Point(2, 2));
                        Mat ekernel = Imgproc.getStructuringElement(Imgproc.MORPH_ERODE, new org.opencv.core.Size(5, 5), new Point(2, 2));
                        Imgproc.erode(fgMask, fgMask, ekernel);
                        Imgproc.dilate(fgMask, fgMask, dkernel);
                        fgbmp = Bitmap.createBitmap(fgMask.width(), fgMask.height(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(fgMask, fgbmp);
                        fgbmp = Bitmap.createScaledBitmap(fgbmp, fgView.getWidth(), fgView.getHeight(), true);
                        //long t4 = System.currentTimeMillis()5
                        //t4 = System.currentTimeMillis();
                        //Log.i(TAG, "timing t34:"+(t4-t3));
                        TimeRunnable tr = new TimeRunnable() {
                            @Override
                            public void run() {
                                //tv.setText(predictedClass);
                                //String ss = "tview";
                                //tv.setText(ss);
                                //Imgproc.cvtColor(frame, frame, Imgproc.COLOR_GRAY2RGBA, 4);
                                //long tui1 = System.currentTimeMillis();

                                bgView.setImageBitmap(bgbmp);
                                bgView.invalidate();
                                bgView.setVisibility(View.VISIBLE);

                                //Imgproc.cvtColor(fgMask, fgMask, Imgproc.COLOR_GRAY2RGBA, 4);

                                fgView.setImageBitmap(fgbmp);
                                fgView.invalidate();
                                fgView.setVisibility(View.VISIBLE);
                                //long tui2 = System.currentTimeMillis();
                                //Log.i(TAG, "timing tui:"+(tui2-tui1)+ " subtractouttime:"+(tui2-t));
//                                Toast.makeText(ClassifyCamera.this,
//                                        fullname,
//                                        Toast.LENGTH_SHORT).show();
                            }};
                        //tr.setTime(t0);
                        runOnUiThread( tr);

                        //! [display_frame_number]
                        // get the frame number and write it on the current frame
                        //Imgproc.rectangle(frame, new Point(10, 2), new Point(100, 20), new Scalar(255, 255, 255), -1);
                        /*String frameNumberString = String.format("%d", (int)capture.get(Videoio.CAP_PROP_POS_FRAMES));
                        Imgproc.putText(frame, frameNumberString, new Point(15, 15), Core.FONT_HERSHEY_SIMPLEX, 0.5,
                                new Scalar(0, 0, 0));*/
                        //! [display_frame_number]


                        // TODO: use these for proper image processing on different formats.
//                        int rowStride = image.getPlanes()[1].getRowStride();
//                        int pixelStride = image.getPlanes()[1].getPixelStride();
//                        Ybuffer.rewind();
//                        Ubuffer.rewind();
//                        Vbuffer.rewind();
//                        byte[] Y = new byte[Ybuffer.capacity()];
//                        byte[] U = new byte[Ubuffer.capacity()];
//                        byte[] V = new byte[Vbuffer.capacity()];
//                        Ybuffer.get(Y);
//                        Ubuffer.get(U);
//                        Vbuffer.get(V);
                        //hasChanged = false;
                        //double rate = 0.0;
                        ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
                        Mat hierarchy = new Mat();
                        Imgproc.findContours(fgMask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

                        TreeMap<Double, Integer> tm=new TreeMap<Double, Integer>();
                        Iterator<MatOfPoint> iterator = contours.iterator();
                        double area_sum = 0;
                        int i =0;
                        //long t5 = System.currentTimeMillis();
                        //Log.i(TAG, "timing t45:"+(t5-t4));
                        while(iterator.hasNext()) {
                            MatOfPoint contour = iterator.next();
                            double area = Imgproc.contourArea(contour);
                            Log.i(TAG, "area:" + area + " area_sum:" + area_sum);
                            if (area > 50) {
                                area_sum += area;
                                tm.put(area, i);
                            }
                            i++;
                        }
                        //long t6 = System.currentTimeMillis();
                        //Log.i(TAG, "timing t56:"+(t6-t5)+" curenttotal:"+(t6-t0));
                        Log.i(TAG, "area_sum/nMaskPixelNum:"+1.0*area_sum/nMaskPixelNum);
                        if (1.0*area_sum/nMaskPixelNum < 0.01){
                            m_qAreaSum.clear();
                            m_qContourCenter.clear();
                            processing = false;
                            return;
                        }

                        Iterator<Integer> it = tm.values().iterator();
                        i=0;
//                        double xs = 0, ys=0;
//                        int npt = 0;
//                        double dst = 0;
//                        while(it.hasNext()){
//                            Integer idx =  it.next();
//                            MatOfPoint c = contours.get(idx);
//                            Moments p = Imgproc.moments(c);
//                            int x = (int) (p.get_m10() / p.get_m00());
//                            int yy = (int) (p.get_m01() / p.get_m00());
//                            dst = Math.sqrt(Math.pow(x-width/2, 2) + Math.pow(yy-height/2, 2));
//                            break;
//                        }
                        double ema = 0.0;

                        if (m_qAreaSum.size() < m_nFrames4BG) {
                            m_qAreaSum.add(area_sum);
                        } else {
                            m_qAreaSum.remove();
                            m_qAreaSum.add(area_sum);
                        }
                        ema = (area_sum - m_fLastEMA) * 2 / (1 + m_qAreaSum.size()) + m_fLastEMA;
                        //double dstEMA = (dst - m_fLastContour2Center)*2/(1+m_qAreaSum.size())+m_fLastContour2Center;
//                        if (ema >= m_fLastEMA) {
//                            m_nUpEMA++;
//                            m_nDownEMA = 0;
//                        } else {
//                            m_nUpEMA = 0;
//                            m_nDownEMA++;
//                        }
//                        m_nUpEMA = min(max(m_nUpEMA, -5), 5);
//                        m_nDownEMA = min(max(m_nDownEMA, -5), 5);
//                        if (area_sum > 100) {
//                            if (dstEMA >= m_fLastContour2Center) {
//                                m_nAwayFrame++;
//                            } else {
//                                m_nAwayFrame--;
//                            }
//                            m_nAwayFrame = min(max(m_nAwayFrame, -5), 5);
//                        }
//                        else{
//                            m_nUpEMA = 0;
//                            m_nDownEMA = 0;
//                            m_nAwayFrame = 0;
//                        }
//                        Log.i(TAG, "m_fLastEMA:"+m_fLastEMA+" ema:"+ema+" m_fLastContour2Center:"+
//                                m_fLastContour2Center+" dstEMA:"+dstEMA+ " m_nAwayFrame:"+m_nAwayFrame+
//                                "m_nDownEMA:"+m_nDownEMA+" area_sum:"+area_sum);
//                        m_fLastContour2Center = dstEMA;
                        double fg_rate = 1.0*ema / nMaskPixelNum;
                        Log.i(TAG,"ema fg_rate:"+fg_rate+" currentrate:"+1.0*area_sum/nMaskPixelNum);
                        if ( fg_rate > 0.2 && fg_rate < 0.001) {
                            processing = false;
                            return;
                        }
                        m_fLastEMA = ema;
//                        if (m_nDownEMA < 3 && (area_sum > nMaskPixelNum*0.1) || m_nAwayFrame > -1|| area_sum > nMaskPixelNum*0.4) {
//                            processing = false;
//                            return;
//                        }

                        fullname="";
                        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED))
                        // 判断是否可以对SDcard进行操作
                        {      // 获取SDCard指定目录下
                            Date c = Calendar.getInstance().getTime();
                            long milliSec = c.getTime();
                            if (milliSec - m_lLastMilliSeconds <1000){
                                processing = false;
                                return;
                            }
                            m_lLastMilliSeconds = milliSec;
                            Log.i(TAG, "Current time => " + c);
                            SimpleDateFormat datef = new SimpleDateFormat("yyyy-MM-dd");
                            SimpleDateFormat timef = new SimpleDateFormat("HH-mm-ss");

                            String formattedDate = datef.format(c);
                            String formattedTime = timef.format(c);
                            //String sdCardDir = "/storage/emulated/0/aaaaaa/" + formattedDate + "/";
                            String sdCardDir = Environment.getExternalStorageDirectory()+ "/aaaaaa/" + formattedDate + "/";
                            //String sdCardDir = m_sExternalRoot + "/Android/data/facebook.f8demo/";

                            File dirFile = new File(sdCardDir);  //目录转化成文件夹
                            if (!dirFile.exists()) {                //如果不存在，那就建立这个文件夹
                                //dirFile .mkdirs();
                                Toast.makeText(ClassifyCamera.this,
                                        (dirFile.mkdirs() ? "Directory has been created" : "Directory not created"),
                                        Toast.LENGTH_SHORT).show();
                            }
                            //文件夹有啦，就可以保存图片啦
                            String fname = formattedTime + "-" + System.currentTimeMillis() + ".jpg";
                            fullname = sdCardDir + fname;
                            Imgcodecs.imwrite(fullname, mBGR);
                            Log.i(TAG, "保存到_sd_指定目录文件夹下_" + fullname);
//                            runOnUiThread(new Runnable() {
//                                @Override
//                                public void run() {
//                                    Toast.makeText(ClassifyCamera.this,
//                                            "saved :" + fullname,
//                                            Toast.LENGTH_SHORT).show();
//                                }
//                            });
                        }
//                        predictedClass = classificationFromCaffe2(h, w, Y, U, V,
//                                rowStride, pixelStride, run_HWC);
//                        predictedClass = predictedClass;  //+ " rate:"+rate;

                        processing = false;
                        //long t7 = System.currentTimeMillis();
                        //Log.i(TAG, "timing t67:"+(t7-t6)+" total:"+(t7-t0));
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
            m_iCameraOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
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
