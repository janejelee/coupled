

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/iterative_inverse.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/lac/sparse_ilu.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/base/tensor_function.h>

namespace Step20
{
  using namespace dealii;
    using namespace numbers;

  namespace data
  {
  	  const int problem_degree = 1;
  	  const int refinement_level = 4;
  	  const int dimension = 2;
  			  
      const double rho_f = 1.0;

      const double top = 1.0;
      const double bottom = 0.0;      
      const double left = 0.0;
      const double right = PI;
     
      const double lambda = 1.0;
      const double k = 1.0;
      const double phi = 1.0;
      
      const double coeff = 1.0;


  }
    
    template <int dim>
    void gravity (const std::vector<Point<dim> > &points,
                          std::vector<Tensor<1, dim> >   &values)
    {

        for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
        {

                values[point_n][0] = 0.0;
                values[point_n][1] = -data::coeff*data::rho_f;
        }
        
    }

    template <int dim>
    void rock_vel (const std::vector<Point<dim> > &points,
                  std::vector<Tensor<1, dim> >   &values)
    {
        
        for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
        {
            
            values[point_n][0] = points[point_n][0]*points[point_n][0] +
                                        points[point_n][1]*points[point_n][1];
            values[point_n][1] = -points[point_n][1]*points[point_n][0];
        }
        
    }

    template <int dim>
    void pressure_flux (const std::vector<Point<dim> > &points,
                   std::vector<Tensor<1, dim> >   &values)
    {
        
        for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
        {
            
            values[point_n][0] = 0.0;//-(points[point_n][1]*points[point_n][1] + 1.0
                                    //- 3* points[point_n][0]*points[point_n][0]);
            values[point_n][1] = -data::rho_f*(1.0-points[point_n][1]*points[point_n][1]); //-2*points[point_n][0]*points[point_n][1];
        }
        
    }
    

    
    
    
  template <int dim>
  class VfProblem
  {
  public:
    VfProblem (const unsigned int degree);
    void run ();

  private:
    void make_grid_and_dofs ();
    void assemble_system ();
    void solve ();
    void compute_errors () const;
    void output_results () const;

    const unsigned int   degree;

    Triangulation<dim>   triangulation;
    FESystem<dim>        fe;
    DoFHandler<dim>      dof_handler;
    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> mass_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
  };

  
  template <int dim>
  class ExactSolution : public Function<dim>
  {
  public:
    ExactSolution () : Function<dim>(dim+1) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };



  template <int dim>
  void
  ExactSolution<dim>::vector_value (const Point<dim> &p,
                                    Vector<double>   &values) const
  {


      values(0) = p[0]*p[0] + p[1]*p[1]; // + data::coeff*(p[1]*p[1] + 1 - 3*p[0]*p[0]);
      values(1) = -p[0]*p[1] - data::rho_f*p[1]*p[1];// + data::coeff*2*p[0]*p[1] - data::coeff*data::rho_f;
  }


    

  template <int dim>
  VfProblem<dim>::VfProblem (const unsigned int degree)
    :
    degree (degree),
    fe (FE_Q<dim>(degree), dim),
    dof_handler (triangulation)
  {}




  template <int dim>
  void VfProblem<dim>::make_grid_and_dofs ()
  {
      {
          std::vector<unsigned int> subdivisions (dim, 1);
          subdivisions[0] = 4;

          const Point<dim> bottom_left = (dim == 2 ?
                                          Point<dim>(data::left,data::bottom) :
                                          Point<dim>(-2,0,-1));
          const Point<dim> top_right   = (dim == 2 ?
                                          Point<dim>(data::right,data::top) :
                                          Point<dim>(0,1,0));

          GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                     subdivisions,
                                                     bottom_left,
                                                     top_right);
      }


      for (typename Triangulation<dim>::active_cell_iterator
           cell = triangulation.begin_active();
           cell != triangulation.end(); ++cell)
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->face(f)->center()[dim-1] == data::top)
                  cell->face(f)->set_all_boundary_ids(1);
              else if (cell->face(f)->center()[dim-1] == data::bottom)
                  cell->face(f)->set_all_boundary_ids(2);


    triangulation.refine_global (data::refinement_level);

    dof_handler.distribute_dofs (fe);
 

    std::cout << "Problem Degree: "
       		  << data::problem_degree
                 << std::endl
   			  << "Refinement level: "
   			  << data::refinement_level
   			  << std::endl
			  << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: "
              << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
 
      
      
      constraints.clear ();
      DoFTools::make_hanging_node_constraints (dof_handler,
                                               constraints);
      constraints.close();
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler,
                                      dsp,
                                      constraints,
                                      /*keep_constrained_dofs = */ true);
      sparsity_pattern.copy_from(dsp);
      
      mass_matrix.reinit (sparsity_pattern);
    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
      
      
      MatrixCreator::create_mass_matrix(dof_handler,
                                        QGauss<dim>(fe.degree+2),
                                        mass_matrix);
  }



  template <int dim>
  void VfProblem<dim>::assemble_system ()
  {
    QGauss<dim>   quadrature_formula(degree+2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    | update_gradients |
                             update_quadrature_points  | update_JxW_values);
      
      
    const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int   n_q_points      = quadrature_formula.size();

    FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       local_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
      
    
      std::vector<Tensor<1,dim>>     gravity_values (n_q_points);
      std::vector<Tensor<1,dim>>     rock_vel_values (n_q_points);
      std::vector<Tensor<1,dim>>     pressure_flux_values (n_q_points);


    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
          
        fe_values.reinit (cell);
        local_matrix = 0;
        local_rhs = 0;

          gravity (fe_values.get_quadrature_points(), gravity_values);
          rock_vel (fe_values.get_quadrature_points(), rock_vel_values);
          pressure_flux (fe_values.get_quadrature_points(), pressure_flux_values);
          
          /*
          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;
              for (unsigned int j=0; j<dofs_per_cell; ++j)
              {
                  const unsigned int
                  component_j = fe.system_to_component_index(j).first;
                  for (unsigned int q_point=0; q_point<n_q_points;
                       ++q_point)
                  {
                      local_matrix(i,j)
                      +=
                      (
                       (fe_values.shape_value (i,q_point) *
                        fe_values.shape_value (j,q_point) )
                       )
                      *
                      fe_values.JxW(q_point);
                  }
              }
          }
          */

          for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
              const unsigned int
              component_i = fe.system_to_component_index(i).first;
              
              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                local_rhs(i) += (fe_values.shape_value(i,q_point) *
                                gravity_values[q_point][component_i] +  // gravity already negative
                                 fe_values.shape_value(i,q_point) * 
                                 rock_vel_values[q_point][component_i]  // rock velocity added
                                 -data::coeff* fe_values.shape_value(i,q_point) *
                                 pressure_flux_values[q_point][component_i] // minus pressure gradient
                                 ) *
                                fe_values.JxW(q_point);
              
          }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               local_matrix(i,j));
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);
          
      }

      constraints.condense (system_matrix);
      constraints.condense (system_rhs);
    
  }





  template <int dim>
  void VfProblem<dim>::solve ()
  {
      system_matrix.copy_from(mass_matrix);
      
      SparseDirectUMFPACK  A_direct;
      A_direct.initialize(system_matrix);
      A_direct.vmult (solution, system_rhs);

  }




  template <int dim>
  void VfProblem<dim>::compute_errors () const
  {


    ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors (triangulation.n_active_cells());

    QTrapez<1>     q_trapez;
    QIterated<dim> quadrature (q_trapez, degree+2);

    VectorTools::integrate_difference (dof_handler, solution, exact_solution,
                                       cellwise_errors, quadrature,
                                       VectorTools::L2_norm);
    const double l2_error = cellwise_errors.l2_norm();


    std::cout << "Errors: L2 = " << l2_error
              << std::endl;
  }



  template <int dim>
  void VfProblem<dim>::output_results () const
  {
      
    std::vector<std::string> solution_names;
    switch (dim)
      {
      case 2:
        solution_names.push_back ("u");
        solution_names.push_back ("v");
        break;

      case 3:
        solution_names.push_back ("u");
        solution_names.push_back ("v");
        solution_names.push_back ("w");
        break;

      default:
        Assert (false, ExcNotImplemented());
      }


    DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, solution_names);

/*    data_out.build_patches (degree+1);

    std::ofstream output ("solution.gmv");
    data_out.write_gmv (output);
    */
    data_out.build_patches ();

    std::ofstream output ("solution.vtk");
    data_out.write_vtk (output);
  }




  template <int dim>
  void VfProblem<dim>::run ()
  {
    make_grid_and_dofs();
    assemble_system ();
    solve ();
    compute_errors ();
    output_results ();
  }
}



int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step20;

      VfProblem<data::dimension> mixed_laplace_problem(data::problem_degree);
      mixed_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
