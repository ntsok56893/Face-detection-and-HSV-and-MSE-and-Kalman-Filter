package com.example.a05.imgfd;

/**
 * Created by 05 on 2018/2/9.
 */

import android.content.Context;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.KalmanFilter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import static org.opencv.core.CvType.CV_32F;

public class DetectActivity extends AppCompatActivity implements
        CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {

    private CameraBridgeViewBase cameraView;
    private CascadeClassifier classifier;
    private Mat mGray;
    private Mat mRgba;
    private Mat prehsvFrame;
    private Mat curhsvFrame;
    private Mat tempsubimg; //delete
    private int mAbsoluteFaceSize = 0;
    private boolean isFrontCamera;

    float[] tM = { 1, 0, 1, 0,
                   0, 1, 0, 1,
                   0, 0, 1, 0,
                   0, 0, 0, 1 } ;
    private KalmanFilter KF;
    private Mat transitionMatrix;
    private Mat measurementMatrix;
    private Mat statePre;
    private Mat processNoiseCov;
    private Mat measurementNoiseCov;
    private Mat id2;
    private Mat prediction;
    private Mat measurement;

    private int frameCount;
    private int frameThreshold;
    private double MSE;
    private int tempX;
    private int tempY;
    private int faceH;
    private int faceW;
    private Point topLPt;
    private Point predictPt;
    private Point predbottRPt;

    static {
        System.loadLibrary("opencv_java3");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        initWindowSettings();
        setContentView(R.layout.activity_detect);
        cameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        cameraView.setCvCameraViewListener(this); // 設置相機監聽
        initClassifier();
        cameraView.enableView();
        cameraView.enableFpsMeter();
        Button switchCamera = (Button) findViewById(R.id.switch_camera);
        switchCamera.setOnClickListener(this); // 切換相機鏡頭，默認後置相機
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.switch_camera:
                cameraView.disableView();
                if (isFrontCamera) {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                    isFrontCamera = false;
                } else {
                    cameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                    isFrontCamera = true;
                }
                cameraView.enableView();
                break;
            default:
        }
    }

    // 初始化窗口設定
    private void initWindowSettings() {
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
    }

    // 初始化人臉級聯分類器
    private void initClassifier() {
        try {
            InputStream is = getResources()
                    .openRawResource(R.raw.haarcascade_frontalface_alt2);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt2.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            classifier = new CascadeClassifier(cascadeFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

        frameCount = 0;
        MSE = 0;
        frameThreshold = 0;
        tempX = 0;
        tempY = 0;

        mGray = new Mat();
        mRgba = new Mat();
        prehsvFrame = new Mat();
        curhsvFrame = new Mat();
        tempsubimg = new Mat();

        KF = new KalmanFilter(4,2,0, CV_32F);

        //set transition matrix
        transitionMatrix=new Mat(4,4, CV_32F,new Scalar(0));
        transitionMatrix.put(0,0,tM);
        KF.set_transitionMatrix(transitionMatrix);

        //set init measurement
        measurementMatrix = new Mat (2,4, CV_32F);
        measurementMatrix.put(0, 0, 1);
        measurementMatrix.put(0, 1, 0);
        measurementMatrix.put(0, 2, 0);
        measurementMatrix.put(0, 3, 0);
        measurementMatrix.put(1, 0, 0);
        measurementMatrix.put(1, 1, 1);
        measurementMatrix.put(1, 2, 0);
        measurementMatrix.put(1, 3, 0);
        KF.set_measurementMatrix(measurementMatrix);

        //Set state matrix
        statePre = new Mat(4,1, CV_32F);
        statePre.put(0, 0, 300);
        statePre.put(1, 0, 200);
        statePre.put(2, 0, 0);
        statePre.put(3, 0, 0);
        KF.set_statePre(statePre);

        //Process noise Covariance matrix
        processNoiseCov=Mat.eye(4,4, CV_32F);
        processNoiseCov=processNoiseCov.mul(processNoiseCov,1e-1);
        KF.set_processNoiseCov(processNoiseCov);

        //Measurement noise Covariance matrix: reliability on our first measurement
        measurementNoiseCov=Mat.eye(2,2, CV_32F);
        measurementNoiseCov=measurementNoiseCov.mul(measurementNoiseCov,1e-1);
        KF.set_measurementNoiseCov(measurementNoiseCov);

        id2=Mat.eye(4,4, CV_32F);
        id2=id2.mul(id2,0.1);
        KF.set_errorCovPost(id2);

        measurement = new Mat (2,1, CV_32F);

        faceH = 0;
        faceW = 0;
        topLPt = new Point(1, 1);
        predictPt = new Point(1, 1);
        predbottRPt = new Point(1, 1);
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
        prehsvFrame.release();
        curhsvFrame.release();
        transitionMatrix.release();
        measurementMatrix.release();
        statePre.release();
        processNoiseCov.release();
        measurementNoiseCov.release();
        id2.release();
        tempsubimg.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // 翻轉矩陣，用來校正前後鏡頭
        if (isFrontCamera) {
            Core.flip(mRgba, mRgba, 1);
            Core.flip(mGray, mGray, 1);
        }
        float mRelativeFaceSize = 0.2f;
        if (mAbsoluteFaceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }
        MatOfRect faces = new MatOfRect();
        if (classifier != null)
            classifier.detectMultiScale(mGray, faces, 1.1, 3, 0,
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        Rect[] facesArray = faces.toArray();
        Scalar faceRectColor = new Scalar(0, 255, 0, 255);
        Scalar faceDectColor = new Scalar(255, 0, 0, 255);

        if(frameCount >= frameThreshold)
        for (Rect faceRect : facesArray) {
            Imgproc.rectangle(mRgba, faceRect.tl(), faceRect.br(), faceRectColor, 3);

            mRgba.submat(faceRect).copyTo(prehsvFrame);
            Imgproc.cvtColor(prehsvFrame, prehsvFrame, Imgproc.COLOR_BGR2HSV);
            Core.extractChannel(prehsvFrame, prehsvFrame, 0);

            faceH = faceRect.height;
            faceW = faceRect.width;
            topLPt.x = (double) faceRect.tl().x;
            topLPt.y = (double) faceRect.tl().y;

            if(!faceRect.empty()){
                frameCount = 0;
                break;
            }
        }

        if(faces.empty()){

            if(!prehsvFrame.empty()){

                prediction = KF.predict();
                tempX = (int) prediction.get(0, 0)[0];
                tempY = (int) prediction.get(1, 0)[0];
                if(tempX > 0 && tempY > 0 && (tempY + faceW) < mRgba.height() && (tempX + faceH) < mRgba.width()){

                    predictPt.x = (int) prediction.get(0, 0)[0];
                    predictPt.y = (int) prediction.get(1, 0)[0];

                    predbottRPt.x = predictPt.x + faceH;
                    predbottRPt.y = predictPt.y + faceW;

                    tempsubimg = mRgba.submat(new Rect((int) predictPt.x,(int) predictPt.y, faceH, faceW));
                    Imgproc.cvtColor(tempsubimg, curhsvFrame, Imgproc.COLOR_BGR2HSV);
                    Core.extractChannel(curhsvFrame, curhsvFrame, 0);

                    Core.absdiff(curhsvFrame, prehsvFrame, curhsvFrame);

                    MSE = Core.sumElems(curhsvFrame).val[0] / curhsvFrame.total(); //將curhsvFrame所有像素加起來除以像素數

                    Imgproc.cvtColor(curhsvFrame, curhsvFrame, Imgproc.COLOR_GRAY2BGRA);

                    curhsvFrame.copyTo(tempsubimg);

                    if ( MSE <= 30) {

                        measurement.put(0, 0, topLPt.x);
                        measurement.put(1, 0, topLPt.y);

                        KF.correct(measurement);
                    }
                    else {
                        measurement.put(0, 0, predictPt.x);
                        measurement.put(1, 0, predictPt.y);

                        KF.correct(measurement);
                    }

                    Imgproc.putText(mRgba, String.valueOf(MSE), new Point(200, 200), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255, 0, 0));
                }

                Imgproc.rectangle(mRgba, predictPt, predbottRPt, faceDectColor, 3);

                //Imgproc.putText(mRgba, String.valueOf(curhsvFrame.get(10, 10)[0]), new Point(200, 200), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255, 0, 0));

                //Imgproc.putText(mRgba, String.valueOf(measurement.get(0, 0)[0]), new Point(200, 200), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255, 0, 0));
                //Imgproc.putText(mRgba, String.valueOf(measurement.get(1, 0)[0]), new Point(200, 500), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(255, 0, 0));
            }

        }else{

            prediction = KF.predict();
            predictPt.x = (int) prediction.get(0, 0)[0];
            predictPt.y = (int) prediction.get(1, 0)[0];

            measurement.put(0, 0, topLPt.x);
            measurement.put(1, 0, topLPt.y);

            predbottRPt.x = predictPt.x + faceH;
            predbottRPt.y = predictPt.y + faceW;
            KF.correct(measurement);

            Imgproc.rectangle(mRgba, predictPt, predbottRPt, faceDectColor, 3);
        }

        frameCount++;

        return mRgba;
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraView != null) {
            cameraView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraView.disableView();
    }
}
