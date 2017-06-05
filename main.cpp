#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

double func_omega(double x, double t)
{
	//return 1 / (1 + exp(x + t));
	return pow((2 + exp((double)7 / 4 * t + 2 * x)), 4);
}

double func_f(double omega)
{
	//return -2 * omega * (1 - omega) * (1 - omega
	return 3 * omega - 12 * sqrt(omega);
}

void tridiagonal_matrix_algorithm(const vector<double> &alpha, const vector<double> &beta,
	const vector<double> &gamma, const vector<double> &b, vector<double> &x)
{
	size_t size = alpha.size();
	vector<double> w_right(size);
	vector<double> v_right(size);
	vector<double> w_left(size);
	vector<double> v_left(size);

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			w_right[0] = -gamma[0] / alpha[0];
			v_right[0] = b[0] / alpha[0];

			for (size_t i = 1; i <= size / 2; ++i)
			{
				w_right[i] = -gamma[i] / (beta[i] * w_right[i - 1] + alpha[i]);
				v_right[i] = (b[i] - beta[i] * v_right[i - 1]) / (beta[i] * w_right[i - 1] + alpha[i]);
			}
		}

		#pragma omp section
		{
			w_left[size - 1] = -beta[size - 1] / alpha[size - 1];
			v_left[size - 1] = b[size - 1] / alpha[size - 1];

			for (size_t i = size - 2; i > size / 2; --i)
			{
				w_left[i] = -beta[i] / (gamma[i] * w_left[i + 1] + alpha[i]);
				v_left[i] = (b[i] - gamma[i] * v_left[i + 1]) / (gamma[i] * w_left[i + 1] + alpha[i]);
			}
		}
	}

	x[size / 2] = (v_right[size / 2] + w_right[size / 2] * v_left[size / 2 + 1]) /
		(1 - w_right[size / 2] * w_left[size / 2 + 1]);

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			for (int i = size / 2 - 1; i >= 0; --i)
			{
				x[i] = w_right[i] * x[i + 1] + v_right[i];
			}
		}
		#pragma omp section
		{
			for (int i = size / 2 + 1; i < size; ++i)
			{
				x[i] = w_left[i] * x[i - 1] + v_left[i];
			}
		}
	}
}

int main()
{
	constexpr double start_x = 0, end_x = 1, start_t = 0, end_t = 1;
	constexpr size_t number_of_steps_x = 50;
	constexpr size_t number_of_steps_t = 5000;
	constexpr double step_x = (end_x - start_x) / (number_of_steps_x - 1);
	constexpr double step_t = (end_t - start_t) / (number_of_steps_t - 1);


	vector<vector<double>> omega(number_of_steps_t, vector<double>(number_of_steps_x));
	vector<double> start_boundary_condition(number_of_steps_t);
	vector<double> end_boundary_condition(number_of_steps_t);

	double x_i = start_x, t_i = start_t;
	for (size_t i = 0; i < number_of_steps_x; ++i)
	{
		omega[0][i] = func_omega(x_i, 0);
		x_i += step_x;
	}
	for (size_t i = 0; i < number_of_steps_t; ++i)
	{
		start_boundary_condition[i] = func_omega(start_x, t_i);
		end_boundary_condition[i] = func_omega(end_x, t_i);

		omega[i][0] = start_boundary_condition[i];
		omega[i][number_of_steps_x - 1] = end_boundary_condition[i];
		t_i += step_t;
	}

	vector<double> alpha(number_of_steps_x - 2, 1 + 2 * step_t / step_x / step_x);
	vector<double> beta(number_of_steps_x - 2, -step_t / step_x / step_x);
	vector<double> gamma(number_of_steps_x - 2, -step_t / step_x / step_x);
	vector<double> b(number_of_steps_x - 2);
	vector<double> x(number_of_steps_x - 2);
	for (size_t j = 1; j < number_of_steps_t; ++j)
	{
		for (size_t i = 1; i < number_of_steps_x - 1; ++i)
			b[i - 1] = omega[j - 1][i] - step_t * func_f(omega[j][i]);
		b[0] += start_boundary_condition[j] * step_t / step_x / step_x;
		b[number_of_steps_x - 3] += end_boundary_condition[j] * step_t / step_x / step_x;

		tridiagonal_matrix_algorithm(alpha, beta, gamma, b, x);
		for (size_t i = 1; i < number_of_steps_x - 1; ++i)
			omega[j][i] = x[i - 1];
	}
	ofstream result_file("result.txt");
	result_file << "{";
	for (size_t i = 0; i < number_of_steps_x; ++i)
	{
		for (size_t j = 0; j < number_of_steps_t; ++j)
		{
			result_file << "{" << start_x + step_x * i << "," << start_t + step_t * j << "," << fixed << omega[j][i];

			if (i == number_of_steps_x - 1 && j == number_of_steps_t - 1)
				result_file << "}";
			else 
				result_file << "},";
		}
	}
	result_file << "}";
	return 0;
}