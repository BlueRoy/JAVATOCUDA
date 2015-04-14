import java.lang.*;

public class J2C {
	public static int SIZE = 10;
	
	static{
		System.loadLibrary("j2c");
	}
	public native int writetocudac(int a, int b, int c);
	public static void main(String[] args){
		System.out.println("Hello C through JNI!");
		
	    J2C m = new J2C();

	    int a = 1;
	    int b = 2;
	    int c = 3;
	   
	    System.out.println("J calling C.");
		
	    int retVal = m.writetocudac(a, b, c);
	    System.out.println("J:"+ retVal);
	    System.out.println();
	}
}
