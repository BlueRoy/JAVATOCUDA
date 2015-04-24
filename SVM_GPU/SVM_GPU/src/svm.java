import java.io.*;
import java.util.*;

public class svm {
	
	static final String svm_type_table[] =
		{
			"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",
		};

		static final String kernel_type_table[]=
		{
			"linear","polynomial","rbf","sigmoid","precomputed"
		};

		
	private static svm_print_interface svm_print_stdout = new svm_print_interface()
	{
		public void print(String s)
		{
			System.out.print(s);
			System.out.flush();
		}
	};
	
	private static svm_print_interface svm_print_string = svm_print_stdout;


	public static String svm_check_parameter(svm_problem prob, svm_parameter param)
	{
		// svm_type

		int svm_type = param.svm_type;
		if(svm_type != svm_parameter.C_SVC &&
		   svm_type != svm_parameter.NU_SVC &&
		   svm_type != svm_parameter.ONE_CLASS &&
		   svm_type != svm_parameter.EPSILON_SVR &&
		   svm_type != svm_parameter.NU_SVR)
		return "unknown svm type";

		// kernel_type, degree
	
		int kernel_type = param.kernel_type;
		if(kernel_type != svm_parameter.LINEAR &&
		   kernel_type != svm_parameter.POLY &&
		   kernel_type != svm_parameter.RBF &&
		   kernel_type != svm_parameter.SIGMOID &&
		   kernel_type != svm_parameter.PRECOMPUTED)
			return "unknown kernel type";

		if(param.gamma < 0)
			return "gamma < 0";

		if(param.degree < 0)
			return "degree of polynomial kernel < 0";

		// cache_size,eps,C,nu,p,shrinking

		if(param.cache_size <= 0)
			return "cache_size <= 0";

		if(param.eps <= 0)
			return "eps <= 0";

		if(svm_type == svm_parameter.C_SVC ||
		   svm_type == svm_parameter.EPSILON_SVR ||
		   svm_type == svm_parameter.NU_SVR)
			if(param.C <= 0)
				return "C <= 0";

		if(svm_type == svm_parameter.NU_SVC ||
		   svm_type == svm_parameter.ONE_CLASS ||
		   svm_type == svm_parameter.NU_SVR)
			if(param.nu <= 0 || param.nu > 1)
				return "nu <= 0 or nu > 1";

		if(svm_type == svm_parameter.EPSILON_SVR)
			if(param.p < 0)
				return "p < 0";

		if(param.shrinking != 0 &&
		   param.shrinking != 1)
			return "shrinking != 0 and shrinking != 1";

		if(param.probability != 0 &&
		   param.probability != 1)
			return "probability != 0 and probability != 1";

		if(param.probability == 1 &&
		   svm_type == svm_parameter.ONE_CLASS)
			return "one-class SVM probability output not supported yet";
		
		// check whether nu-svc is feasible
	
		if(svm_type == svm_parameter.NU_SVC)
		{
			int l = prob.l;
			int max_nr_class = 16;
			int nr_class = 0;
			int[] label = new int[max_nr_class];
			int[] count = new int[max_nr_class];

			int i;
			for(i=0;i<l;i++)
			{
				int this_label = (int)prob.y[i];
				int j;
				for(j=0;j<nr_class;j++)
					if(this_label == label[j])
					{
						++count[j];
						break;
					}

				if(j == nr_class)
				{
					if(nr_class == max_nr_class)
					{
						max_nr_class *= 2;
						int[] new_data = new int[max_nr_class];
						System.arraycopy(label,0,new_data,0,label.length);
						label = new_data;
						
						new_data = new int[max_nr_class];
						System.arraycopy(count,0,new_data,0,count.length);
						count = new_data;
					}
					label[nr_class] = this_label;
					count[nr_class] = 1;
					++nr_class;
				}
			}

			for(i=0;i<nr_class;i++)
			{
				int n1 = count[i];
				for(int j=i+1;j<nr_class;j++)
				{
					int n2 = count[j];
					if(param.nu*(n1+n2)/2 > Math.min(n1,n2))
						return "specified nu is infeasible";
				}
			}
		}

		return null;
	}
	
	public static void svm_set_print_string_function(svm_print_interface print_func)
	{
		if (print_func == null)
			svm_print_string = svm_print_stdout;
		else 
			svm_print_string = print_func;
	}


	public static void svm_save_model(String model_file_name, svm_model model) throws IOException
	{
		DataOutputStream fp = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(model_file_name)));

		svm_parameter param = model.param;

		fp.writeBytes("svm_type "+svm_type_table[param.svm_type]+"\n");
		fp.writeBytes("kernel_type "+kernel_type_table[param.kernel_type]+"\n");

		if(param.kernel_type == svm_parameter.POLY)
			fp.writeBytes("degree "+param.degree+"\n");

		if(param.kernel_type == svm_parameter.POLY ||
		   param.kernel_type == svm_parameter.RBF ||
		   param.kernel_type == svm_parameter.SIGMOID)
			fp.writeBytes("gamma "+param.gamma+"\n");

		if(param.kernel_type == svm_parameter.POLY ||
		   param.kernel_type == svm_parameter.SIGMOID)
			fp.writeBytes("coef0 "+param.coef0+"\n");

		int nr_class = model.nr_class;
		int l = model.l;
		fp.writeBytes("nr_class "+nr_class+"\n");
		fp.writeBytes("total_sv "+l+"\n");
	
		{
			fp.writeBytes("rho");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fp.writeBytes(" "+model.rho[i]);
			fp.writeBytes("\n");
		}
	
		if(model.label != null)
		{
			fp.writeBytes("label");
			for(int i=0;i<nr_class;i++)
				fp.writeBytes(" "+model.label[i]);
			fp.writeBytes("\n");
		}

		if(model.probA != null) // regression has probA only
		{
			fp.writeBytes("probA");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fp.writeBytes(" "+model.probA[i]);
			fp.writeBytes("\n");
		}
		if(model.probB != null) 
		{
			fp.writeBytes("probB");
			for(int i=0;i<nr_class*(nr_class-1)/2;i++)
				fp.writeBytes(" "+model.probB[i]);
			fp.writeBytes("\n");
		}

		if(model.nSV != null)
		{
			fp.writeBytes("nr_sv");
			for(int i=0;i<nr_class;i++)
				fp.writeBytes(" "+model.nSV[i]);
			fp.writeBytes("\n");
		}

		fp.writeBytes("SV\n");
		double[][] sv_coef = model.sv_coef;
		svm_node[][] SV = model.SV;

		for(int i=0;i<l;i++)
		{
			for(int j=0;j<nr_class-1;j++)
				fp.writeBytes(sv_coef[j][i]+" ");

			svm_node[] p = SV[i];
			if(param.kernel_type == svm_parameter.PRECOMPUTED)
				fp.writeBytes("0:"+(int)(p[0].value));
			else	
				for(int j=0;j<p.length;j++)
					fp.writeBytes(p[j].index+":"+p[j].value+" ");
			fp.writeBytes("\n");
		}

		fp.close();
	}

}
