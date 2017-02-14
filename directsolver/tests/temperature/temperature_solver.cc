/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2013
 */
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/tensor_function.h>


#include <fstream>
#include <iostream>
namespace Step26
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const int dimension = 2;
        const int degree = 2 ;
        const double top = 1.0;
        const double bottom = 0.0;
        const double right = PI;
        
        const double sill_time = 0.1;
        const double sill_temp = 1000.0;
        const double temp_top = 0.0;
        const double kappa = 10.0;
        const double heat_flux = -1.0;
        const double timestep = 0.01;
        const double T_0 = 1.0;
        
        
        const double final_time = 15*timestep;
        const double error_time = 13;
    
        
    }
    
    
    template<int dim>
    class HeatEquation
    {
    public:
        HeatEquation (const unsigned int degree);
        void run();
    private:
        void setup_system();
        void solve_time_step();
        void output_results() const;
        void error_analysis();
        void refine_mesh (const unsigned int min_grid_level,
                          const unsigned int max_grid_level);
        
        const unsigned int degree;
        Triangulation<dim>   triangulation;
        FE_Q<dim>            fe;
        DoFHandler<dim>      dof_handler;
        ConstraintMatrix     constraints;
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> advection_matrix;
        SparseMatrix<double> system_matrix;
        Vector<double>       solution;
        Vector<double>       old_solution;
        Vector<double>       system_rhs;
        Vector<double>       system_rhs_mass;
        double               time;
        double               time_step;
        unsigned int         timestep_number;
        const double         theta;
    };
    
    
    template<int dim>
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide ()
        :
        Function<dim>(),
        period (0.2)
        {}
        virtual double value (const Point<dim> &p,
                              const unsigned int component = 0) const;
    private:
        const double period;
    };

    
    
        template<int dim>
        double RightHandSide<dim>::value (const Point<dim> &p,
                                          const unsigned int component) const
        {
            const double time = this->get_time();
            
            return (data::kappa *((PI/data::right)*(PI/data::right) +
                                   (PI/data::top)*(PI/data::top)) - PI)*
            (std::cos(PI*p[0]/data::right)*std::sin(PI*p[1]/data::top)*
             std::exp(- PI * time)) +
            (std::cos(PI *p[0]/data::right) * std::cos(PI *p[1]/data::top)*
             std::exp(-PI * time) * PI/p[1]);
            
            
            /*Assert (component == 0, ExcInternalError());
            Assert (dim == 2, ExcNotImplemented());
            const double time = this->get_time();
            if ((time >= 0.15) && (time <= 0.2))
            {
                if ((p[1] > 0.4*data::top) && (p[1] < 0.6*data::top))
                    return data::sill_temp;
                else
                    return 0;
            }
            else
                return 0;
             */
        }
    
        
    template<int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        virtual double value (const Point<dim>  &p,
                              const unsigned int component = 0) const;
    };
    
    
    template<int dim>
    double BoundaryValues<dim>::value (const Point<dim> &/*p*/,
                                       const unsigned int component) const
    {
        Assert(component == 0, ExcInternalError());
        const double time = this->get_time();
        return time;
    }
    
    
    template<int dim>
    class HeatFluxValues : public Function<dim>
    {
    public:
        HeatFluxValues () : Function<dim>() {}
        virtual double value (const Point<dim>  &p,
                              const unsigned int component = 0) const;
    };
    
    
    template<int dim>
    double HeatFluxValues<dim>::value (const Point<dim> &p,
                                       const unsigned int component) const
    {
        
        const double time = this->get_time();
        
        return (PI / data::top)*std::exp(-PI*time)*std::cos(PI*p[0]/data::right
                                                                     ) ;
    }
    
    
    
    template <int dim>
      class AdvectionField : public TensorFunction<1,dim>
      {
      public:
        AdvectionField () : TensorFunction<1,dim> () 
		{}
        virtual Tensor<1,dim> value (const Point<dim> &p) const;
        
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<1,dim> >    &values) const;
        


      };
    
      template <int dim>
      Tensor<1,dim>
      AdvectionField<dim>::value (const Point<dim> &p) const
      {
        Point<dim> value;
        value[0] = 0.0;
          value[1] = 1.0;
        
        return value;
      }
      
      template <int dim>
      void
      AdvectionField<dim>::value_list (const std::vector<Point<dim> > &points,
                                       std::vector<Tensor<1,dim> >    &values) const
      {
        Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
        
        for (unsigned int i=0; i<points.size(); ++i)
          values[i] = AdvectionField<dim>::value (points[i]);
      }

    
    
       template <int dim>
         class InitialFunction : public Function<dim>
        {
         public:
             InitialFunction () : Function<dim>() {}
             virtual double value (const Point<dim>   &p,
                                   const unsigned int  component = 0) const;
         };
         
         template <int dim>
         double InitialFunction<dim>::value (const Point<dim>  &p,
                                             const unsigned int /*component*/) const
         {
             const double time = this->get_time();
             return (std::cos(PI*p[0]/data::right)*std::sin(PI*p[1]/data::top)
                     ) + time;
         }
    
    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
        ExactSolution () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExactSolution<dim>::value (const Point<dim>  &p,
                                        const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        
        return  (std::cos(PI * p[0] / data::right)* std::sin(PI*p[1]/data::top) ) *
                std::exp(-PI*time) + time;

    }
    
    
    
    
	 
    template<int dim>
    HeatEquation<dim>::HeatEquation (const unsigned int degree)
    :
    degree (degree),
    fe(degree+1),
    dof_handler(triangulation),
    time_step(data::timestep),
    theta(0.5)
    {}
    
    
    template<int dim>
    void HeatEquation<dim>::setup_system()
    {
        dof_handler.distribute_dofs(fe);
        /*
        std::cout << std::endl
        << "==========================================="
        << std::endl
        << "Number of active cells: " << triangulation.n_active_cells()
        << std::endl
        << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl
        << std::endl;*/
        
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
        
        mass_matrix.reinit(sparsity_pattern);
        laplace_matrix.reinit(sparsity_pattern);
        advection_matrix.reinit(sparsity_pattern);
        system_matrix.reinit(sparsity_pattern);
        
        solution.reinit(dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
        system_rhs_mass.reinit(dof_handler.n_dofs());
        
        
        
        MatrixCreator::create_mass_matrix(dof_handler,
                                          QGauss<dim>(fe.degree+2),
                                          mass_matrix);
        
                QGauss<dim>  quadrature_formula(degree+2);
                QGauss<dim-1> face_quadrature_formula(degree+2);
                
                
                FEValues<dim> fe_values (fe, quadrature_formula,
                                          update_values    |  update_gradients |
                                         update_quadrature_points  |  update_JxW_values);
                FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                                  update_values         | update_quadrature_points  |
                                                  update_normal_vectors | update_JxW_values);
                
                
                
                 const unsigned int   dofs_per_cell = fe.dofs_per_cell;
                 const unsigned int   n_q_points    = quadrature_formula.size();
                 const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
                 
                 const AdvectionField<dim> advection_field;
                 std::vector<Tensor<1,dim> > advection_directions (n_q_points);
     
                 advection_field.value_list (fe_values.get_quadrature_points(),
                                             advection_directions);
        
        
                const HeatFluxValues<dim> heatflux;
                std::vector<double> heatflux_values (n_face_q_points);
        
        
                 FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
                 FullMatrix<double>   cell_advection_matrix (dofs_per_cell, dofs_per_cell);
                 Vector<double>       cell_rhs (dofs_per_cell);
                 
                 std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
                 
                typename DoFHandler<dim>::active_cell_iterator
                 cell = dof_handler.begin_active(),
                 endc = dof_handler.end();
                 for (; cell!=endc; ++cell)
                 {
                     cell_matrix = 0;
                     cell_advection_matrix = 0;
                     cell_rhs = 0;
                     
                     fe_values.reinit (cell);
                     
                     for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
                     {
                         
                             for (unsigned int i=0; i<dofs_per_cell; ++i)
                         {
                             for (unsigned int j=0; j<dofs_per_cell; ++j)
                             {
                                 cell_matrix(i,j) += ( (data::kappa *
                                                      fe_values.shape_grad(i,q_index) *
                                                      fe_values.shape_grad(j,q_index)) *
                                                      fe_values.JxW(q_index));
                                 
                                 
                                 
                                 
                             
                             	 cell_advection_matrix(i,j) += 0.0*
													  (advection_directions[q_index] * 
														fe_values.shape_grad(i,q_index) *
														fe_values.shape_value(j,q_index)
															  )*fe_values.JxW(q_index);
                                 
                             	 
                     }
                         }
                     }

                     
                     cell->get_dof_indices (local_dof_indices);
                     for (unsigned int i=0; i<dofs_per_cell; ++i)
                     {
                         for (unsigned int j=0; j<dofs_per_cell; ++j)
                             laplace_matrix.add (local_dof_indices[i],
                                                 local_dof_indices[j],
                                                 cell_matrix(i,j));
                         
                        for (unsigned int j=0; j<dofs_per_cell; ++j)
                         {
                              advection_matrix.add (local_dof_indices[i],
                                                    local_dof_indices[j],
                                                    cell_advection_matrix(i,j));
                         
                         }
                         
                     }
                 }
        
        

    }
    
    

    
    template<int dim>
    void HeatEquation<dim>::solve_time_step()
    {
        SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
        SolverCG<> cg(solver_control);
        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.0);
        
        cg.solve(system_matrix, solution, system_rhs,
                 preconditioner);
        
        constraints.distribute(solution);
        /*
        std::cout << "     " << solver_control.last_step()
        << " CG iterations." << std::endl;
         
         */
    }
    
    
    template<int dim>
    void HeatEquation<dim>::output_results() const
    {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "U");
        data_out.build_patches();
        const std::string filename = "solution-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output(filename.c_str());
        data_out.write_vtk(output);
    }
    
    
    
    template <int dim>
    void HeatEquation<dim>::refine_mesh (const unsigned int min_grid_level,
                                         const unsigned int max_grid_level)
    {
        /*
        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(fe.degree+1),
                                            typename FunctionMap<dim>::type(),
                                            solution,
                                            estimated_error_per_cell);
        GridRefinement::refine_and_coarsen_fixed_fraction (triangulation,
                                                           estimated_error_per_cell,
                                                           0.6, 0.4);
        if (triangulation.n_levels() > max_grid_level)
            for (typename Triangulation<dim>::active_cell_iterator
                 cell = triangulation.begin_active(max_grid_level);
                 cell != triangulation.end(); ++cell)
                cell->clear_refine_flag ();
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active(min_grid_level);
             cell != triangulation.end_active(min_grid_level); ++cell)
            cell->clear_coarsen_flag ();
        
        SolutionTransfer<dim> solution_trans(dof_handler);
        Vector<double> previous_solution;
        previous_solution = solution;
        
        triangulation.prepare_coarsening_and_refinement();
        solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
        triangulation.execute_coarsening_and_refinement ();
        
        setup_system ();
        
        solution_trans.interpolate(previous_solution, solution);
        */
        triangulation.refine_global(1);
        
        constraints.distribute (solution);
        
    }
    
    
    template<int dim>
    void HeatEquation<dim>::run()
    {
        const unsigned int initial_global_refinement = 2;
        const unsigned int n_adaptive_pre_refinement_steps = 4;
        
        {
            std::vector<unsigned int> subdivisions (dim, 1);
            subdivisions[0] = 4;
            
            const Point<dim> bottom_left = (dim == 2 ?
                                            Point<dim>(0,data::bottom) :
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
        
        triangulation.refine_global (initial_global_refinement+2);
        
        setup_system();
        
        
        
        unsigned int pre_refinement_step = 0;
        Vector<double> tmp;
        Vector<double> forcing_terms;
        
        start_time_iteration:
		
        tmp.reinit (solution.size());
        forcing_terms.reinit (solution.size());
        
        VectorTools::interpolate(dof_handler,
                                 InitialFunction<dim>(),
                                 old_solution);
        
        solution = old_solution;
        
        timestep_number = 0;
        time            = 0;      
        
        output_results();
        while (time <= data::final_time)
        {
            time += time_step;
            ++timestep_number;
            
            // std::cout << "Time step " << timestep_number << " at t=" << time
            // << std::endl;
            
            // ///////////
            
            mass_matrix.vmult(system_rhs, old_solution);
        
            laplace_matrix.vmult(tmp, old_solution);
            system_rhs.add(-(1 - theta) * time_step, tmp);
            
            advection_matrix.vmult(tmp, old_solution);
            system_rhs.add(-(1 - theta) * time_step, tmp);
            
            
            system_rhs_mass.reinit(dof_handler.n_dofs());
            
            
            QGauss<dim-1> face_quadrature_formula(2);
            
            FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                              update_values         | update_quadrature_points  |
                                              update_normal_vectors | update_JxW_values);
            
            const unsigned int   n_face_q_points = face_quadrature_formula.size();
            
            
            const HeatFluxValues<dim> heatflux;
            std::vector<double> heatflux_values (n_face_q_points);
            
            
           const unsigned int   dofs_per_cell = fe.dofs_per_cell;
            Vector<double>       cell_rhs (dofs_per_cell);
            
           std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
            
            typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
            for (; cell!=endc; ++cell)
            {
                cell_rhs = 0;
                
                
                
                // Heat flux for right hand side
                for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                    if (cell->face(face_number)->at_boundary()
                        &&
                        (cell->face(face_number)->boundary_id() == 2))
                    {
                        fe_face_values.reinit (cell, face_number);
                        
                        heatflux.value_list (fe_face_values.get_quadrature_points(),
                                             heatflux_values);
                        
                        for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        {
                            // see step-7
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                                cell_rhs(i) += -( heatflux_values[q_point]*
                                                 fe_face_values.shape_value(i,q_point) *
                                                 fe_face_values.JxW(q_point));
                            // std::cout << heatflux_values[q_point] << std::endl;
                        }
                    }
                
                cell->get_dof_indices (local_dof_indices);
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    
                    system_rhs_mass(local_dof_indices[i]) += cell_rhs(i);
                }
            }
            
            
            // system_rhs += system_rhs_mass; // Neumann conditions
            
            //
            
            RightHandSide<dim> rhs_function;
            rhs_function.set_time(time);
            
            VectorTools::create_right_hand_side(dof_handler,
                                                QGauss<dim>(fe.degree+1),
                                                rhs_function,
                                                tmp);
            
            forcing_terms = tmp;
            forcing_terms *= time_step * theta;
            rhs_function.set_time(time - time_step);
            VectorTools::create_right_hand_side(dof_handler,
                                                QGauss<dim>(fe.degree+1),
                                                rhs_function,
                                                tmp);
            
            forcing_terms.add(time_step * (1 - theta), tmp);
            system_rhs += forcing_terms;
            
            
            // ////////
            
            system_matrix.copy_from(mass_matrix);
            system_matrix.add(theta * time_step, laplace_matrix);
            system_matrix.add(theta * time_step, advection_matrix);
            constraints.condense (system_matrix, system_rhs);
            
            {
                BoundaryValues<dim> boundary_values_function;
                boundary_values_function.set_time(time);
                std::map<types::global_dof_index, double> boundary_values;
                VectorTools::interpolate_boundary_values(dof_handler,
                                                         1,
                                                         boundary_values_function,
                                                         boundary_values);
                
                VectorTools::interpolate_boundary_values(dof_handler,
                                                         2,
                                                         boundary_values_function,
                                                         boundary_values);
                
                

                MatrixTools::apply_boundary_values(boundary_values,
                                                   system_matrix,
                                                   solution,
                                                   system_rhs);
                
                
            HeatFluxValues<dim> heatflux_function;
            heatflux_function.set_time(time);
                
                
            
            }
            
            solve_time_step();
            
            //if (timestep_number == data::error_time)
                {
            ExactSolution<dim> exact_solution;
                exact_solution.set_time(time);
            Vector<double> difference_per_cell (triangulation.n_active_cells());
                    std::cout << triangulation.n_active_cells() << std::endl;
            
            const QTrapez<1>     q_trapez;
            const QIterated<dim> quadrature (q_trapez, degree+2);
            
            VectorTools::integrate_difference (dof_handler,
                                               solution,
                                               exact_solution,
                                               difference_per_cell,
                                               quadrature,
                                               VectorTools::L2_norm);
            
            const double L2_error = difference_per_cell.l2_norm();
            
            std::cout << "Errors: L2 = " << L2_error << std::endl;
                }
            
            output_results();
            
       /*
            
            if ((timestep_number == 1) &&
                (pre_refinement_step < n_adaptive_pre_refinement_steps))
            {
                //refine_mesh (initial_global_refinement,
                  //           initial_global_refinement + n_adaptive_pre_refinement_steps);
                ++pre_refinement_step;
                tmp.reinit (solution.size());
                forcing_terms.reinit (solution.size());
                std::cout << std::endl;
                goto start_time_iteration;
            }
            else if ((timestep_number > 0) && (timestep_number % 5 == 0))
            {
                //refine_mesh (initial_global_refinement,
                  //           initial_global_refinement + n_adaptive_pre_refinement_steps);
                tmp.reinit (solution.size());
                forcing_terms.reinit (solution.size());
            }
            
            */
            old_solution = solution;
        }
    }

}


int main()
{
    try
    {
        using namespace dealii;
        using namespace Step26;
        HeatEquation<data::dimension> heat_equation_solver(data::degree);
        heat_equation_solver.run();
        
        }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Exception on processing: " << std::endl << exc.what()
        << std::endl << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
        << "----------------------------------------------------"
        << std::endl;
        std::cerr << "Unknown exception!" << std::endl << "Aborting!"
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
        return 1;
    }
    return 0;
}
