 /*
 * solve two laplaceś equation in one code and be able to use the solutions
 *
 *
 *
 *
 */
 

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>


#include <deal.II/base/quadrature_lib.h>

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>

using namespace dealii;



class Step3
{//
public:
  Step3 ();

  void run ();


private:
  void make_grid ();
  void setup_system ();
  void assemble_system ();
  void solve ();
  void third_solution ();
  void output_results () const;

  Triangulation<2>     triangulation;
  FE_Q<2>              fe1;
  FE_Q<2>              fe2;
  FE_Q<2>              fe3;
  DoFHandler<2>        dof_handler1;
  DoFHandler<2>        dof_handler2;
  DoFHandler<2>        dof_handler3;

  SparsityPattern      sparsity_pattern1;
  SparseMatrix<double> system_matrix1;
  SparsityPattern      sparsity_pattern2;
  SparseMatrix<double> system_matrix2;
  
  ConstraintMatrix	   constraints;
  
  Vector<double>       solution1;
  Vector<double>	   solution1_new;
  Vector<double>       system_rhs1;
  
  Vector<double>       solution2;
  Vector<double>       system_rhs2;
  
  Vector<double>       solution3;
};


Step3::Step3 ()
  :
  fe1 (1),
  fe2 (2),
  fe3 (2),  
  dof_handler1 (triangulation),
  dof_handler2 (triangulation),
  dof_handler3 (triangulation)
{}



void Step3::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (5);

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
}




void Step3::setup_system ()
{
  dof_handler1.distribute_dofs (fe1);
  std::cout << "Number of degrees of freedom for u1: "
            << dof_handler1.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp1(dof_handler1.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler1, dsp1);
  sparsity_pattern1.copy_from(dsp1);

  system_matrix1.reinit (sparsity_pattern1);

  solution1.reinit (dof_handler1.n_dofs());
  system_rhs1.reinit (dof_handler1.n_dofs());
  
  //////////////////////////////////////////////////////////////////
  
  dof_handler2.distribute_dofs (fe2);
  std::cout << "Number of degrees of freedom for u2: "
            << dof_handler2.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp2(dof_handler2.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler2, dsp2);
  sparsity_pattern2.copy_from(dsp2);

  system_matrix2.reinit (sparsity_pattern2);

  solution2.reinit (dof_handler2.n_dofs());
  system_rhs2.reinit (dof_handler2.n_dofs());

  
}



void Step3::assemble_system ()
{
  QGauss<2>  quadrature_formula(2);
  FEValues<2> fe_values1 (fe1, quadrature_formula,
                         update_values | update_gradients | update_JxW_values);
  FEValues<2> fe_values2 (fe2, quadrature_formula,
                         update_values | update_gradients | update_JxW_values);

  const unsigned int   dofs_per_cell1 = fe1.dofs_per_cell;
  const unsigned int   dofs_per_cell2 = fe2.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix1 (dofs_per_cell1, dofs_per_cell1);
  Vector<double>       cell_rhs1 (dofs_per_cell1);
  FullMatrix<double>   cell_matrix2 (dofs_per_cell2, dofs_per_cell2);
  Vector<double>       cell_rhs2 (dofs_per_cell2);

  std::vector<types::global_dof_index> local_dof_indices1 (dofs_per_cell1);
  std::vector<types::global_dof_index> local_dof_indices2 (dofs_per_cell2);

  /////////////////////////////////////////////////////
  
  DoFHandler<2>::active_cell_iterator cell1 = dof_handler1.begin_active();
  DoFHandler<2>::active_cell_iterator endc1 = dof_handler1.end();
  for (; cell1!=endc1; ++cell1)
    {
      fe_values1.reinit (cell1);

      cell_matrix1 = 0;
      cell_rhs1 = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int i=0; i<dofs_per_cell1; ++i)
            for (unsigned int j=0; j<dofs_per_cell1; ++j)
              cell_matrix1(i,j) += (fe_values1.shape_grad (i, q_index) *
                                   fe_values1.shape_grad (j, q_index) *
                                   fe_values1.JxW (q_index));

          for (unsigned int i=0; i<dofs_per_cell1; ++i)
            cell_rhs1(i) += (fe_values1.shape_value (i, q_index) *
                            1 *
                            fe_values1.JxW (q_index));
        }
      cell1->get_dof_indices (local_dof_indices1);

      for (unsigned int i=0; i<dofs_per_cell1; ++i)
        for (unsigned int j=0; j<dofs_per_cell1; ++j)
          system_matrix1.add (local_dof_indices1[i],
                             local_dof_indices1[j],
                             cell_matrix1(i,j));

      for (unsigned int i=0; i<dofs_per_cell1; ++i)
        system_rhs1(local_dof_indices1[i]) += cell_rhs1(i);
    }
  

  std::map<types::global_dof_index,double> boundary_values1;
  VectorTools::interpolate_boundary_values (dof_handler1,
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values1);
  MatrixTools::apply_boundary_values (boundary_values1,
                                      system_matrix1,
                                      solution1,
                                      system_rhs1);
  
  ////////////////////////////////////////////////////
  
  
  DoFHandler<2>::active_cell_iterator cell2 = dof_handler2.begin_active();
  DoFHandler<2>::active_cell_iterator endc2 = dof_handler2.end();
  for (; cell2!=endc2; ++cell2)
    {
      fe_values2.reinit (cell2);

      cell_matrix2 = 0;
      cell_rhs2 = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          for (unsigned int i=0; i<dofs_per_cell2; ++i)
            for (unsigned int j=0; j<dofs_per_cell2; ++j)
              cell_matrix2(i,j) += (fe_values2.shape_grad (i, q_index) *
                                   fe_values2.shape_grad (j, q_index) *
                                   fe_values2.JxW (q_index));

          for (unsigned int i=0; i<dofs_per_cell2; ++i)
            cell_rhs2(i) += (fe_values2.shape_value (i, q_index) *
                            2 *
                            fe_values2.JxW (q_index));
        }
      cell2->get_dof_indices (local_dof_indices2);

      for (unsigned int i=0; i<dofs_per_cell2; ++i)
        for (unsigned int j=0; j<dofs_per_cell2; ++j)
          system_matrix2.add (local_dof_indices2[i],
                             local_dof_indices2[j],
                             cell_matrix2(i,j));

      for (unsigned int i=0; i<dofs_per_cell2; ++i)
        system_rhs2(local_dof_indices2[i]) += cell_rhs2(i);
    }


  std::map<types::global_dof_index,double> boundary_values2;
  VectorTools::interpolate_boundary_values (dof_handler2,
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values2);
  MatrixTools::apply_boundary_values (boundary_values2,
                                      system_matrix2,
                                      solution2,
                                      system_rhs2);

  constraints.close();
}



void Step3::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver1 (solver_control);
  SolverCG<>              solver2 (solver_control);

  solver1.solve (system_matrix1, solution1, system_rhs1,
                PreconditionIdentity());
  solver2.solve (system_matrix2, solution2, system_rhs2,
          PreconditionIdentity());
}

void Step3::third_solution ()
{
	  QGauss<2>  quadrature_formula(2);


	  solution1_new.reinit (dof_handler2.n_dofs());
	  /*
	  std::vector<double> tmp;
	  QGauss<2>  quadrature_formula(2);
	  
	  VectorTools::project (dof_handler3, constraints,
	                          quadrature_formula, solution1,
	                          solution1_new);*/
	  ////////
	  
	  dof_handler3.distribute_dofs(fe3);
	  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
	    KellyErrorEstimator<2>::estimate (dof_handler1,
	                                        QGauss<1>(3),
	                                        typename FunctionMap<2>::type(),
	                                        solution1,
	                                        estimated_error_per_cell);
	    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
	                                                     estimated_error_per_cell,
	                                                     0, 0);

	  
	  triangulation.prepare_coarsening_and_refinement ();
	  SolutionTransfer<2> solution_transfer(dof_handler1);	 
	
	  
	  solution_transfer.prepare_for_coarsening_and_refinement(solution1);
	  triangulation.execute_coarsening_and_refinement();
	    

	  dof_handler1.distribute_dofs(fe3);
	  
	  

	  Vector<double> tmp(dof_handler3.n_dofs());
	  solution_transfer.interpolate(solution1_new, tmp);
	  std::cout << "length tmp: "
        << solution1.size()
        << std::endl;
	  
	  solution3.reinit (dof_handler2.n_dofs());
	  

	  solution3 = tmp;
	
	/*
	  solution3.reinit (dof_handler2.n_dofs());
	  solution3 = solution2;
	  solution3.add (solution1);
	  */
}


void Step3::output_results () const
{
  DataOut<2> data_out1;
  data_out1.attach_dof_handler (dof_handler1);
  data_out1.add_data_vector (solution1, "solution_u1");
  data_out1.build_patches ();

  std::ofstream output1 ("solution_u1.gpl");
  data_out1.write_gnuplot (output1);
  
  DataOut<2> data_out2;
  data_out2.attach_dof_handler (dof_handler2);
  data_out2.add_data_vector (solution2, "solution_u2");
  data_out2.build_patches ();

  std::ofstream output2 ("solution_u2.gpl");
  data_out2.write_gnuplot (output2);
  
  DataOut<2> data_out3;
  data_out3.attach_dof_handler (dof_handler2);
  data_out3.add_data_vector (solution3, "solution_u3");
  data_out3.build_patches ();

  std::ofstream output3 ("solution_u3.gpl");
  data_out3.write_gnuplot (output3); 
}



void Step3::run ()
{
  make_grid ();
  setup_system ();
  assemble_system ();
  solve ();
  third_solution ();
  output_results ();
}



int main ()
{
  deallog.depth_console (2);

  Step3 laplace_problem;
  laplace_problem.run ();

  return 0;
}