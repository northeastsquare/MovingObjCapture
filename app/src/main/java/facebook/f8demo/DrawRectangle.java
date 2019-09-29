package facebook.f8demo;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.view.View;


public class DrawRectangle extends View {
    Paint paint = new Paint();
    int [] m_aPTS={0,0,40,0,40,40,0,40};
    public DrawRectangle(Context context) {
        super(context);
    }


    @Override
    public void onDraw(Canvas canvas) {
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.FILL);
        paint.setStyle(Paint.Style.STROKE);
        //Rect rect = new Rect(20, 56, 200, 112);
        //canvas.drawRect(rect, paint );
        canvas.drawLine(m_aPTS[0],m_aPTS[1], m_aPTS[2], m_aPTS[3], paint);
        canvas.drawLine(m_aPTS[2],m_aPTS[3], m_aPTS[4], m_aPTS[5], paint);
        canvas.drawLine(m_aPTS[4],m_aPTS[5], m_aPTS[6], m_aPTS[7], paint);
        canvas.drawLine(m_aPTS[6],m_aPTS[7], m_aPTS[0], m_aPTS[1], paint);
    }
    public void setPTS(int [] pts){
        m_aPTS = pts;
        invalidate();
    }

}
