/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
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
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/lac/sparse_direct.h>
#include <fstream>
#include <iostream>


namespace Step9
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
        
        const double timestep = 0.01;
        const double final_time = 15*timestep;
        const double error_time = 13;
        
        const int global_refinement_level = 4;
    }
    
    
    template <int dim>
    class AdvectionProblem
    {
    public:
        AdvectionProblem ();
        ~AdvectionProblem ();
        void run ();
        
    private:
        void make_grid_and_dofs ();
        
        void assemble_system ();
        void assemble_rhs ();

        void solve ();
        void error_analysis ();
        
        
        Triangulation<dim>   triangulation;
        DoFHandler<dim>      dof_handler;
        FE_Q<dim>            fe;
        ConstraintMatrix     hanging_node_constraints;
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> system_matrix;
        Vector<double>       solution;
        Vector<double>       system_rhs;
    };
    
    
    
    
    template <int dim>
    class AdvectionField : public TensorFunction<1,dim>
    {
    public:
        AdvectionField () : TensorFunction<1,dim> () {}
        virtual Tensor<1,dim> value (const Point<dim> &p) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<1,dim> >    &values) const;
        DeclException2 (ExcDimensionMismatch,
                        unsigned int, unsigned int,
                        << "The vector has size " << arg1 << " but should have "
                        << arg2 << " elements.");
    };
    template <int dim>
    Tensor<1,dim>
    AdvectionField<dim>::value (const Point<dim> &p) const
    {
        Point<dim> value;
        value[0] = p[0];
        value[1] = -p[1];
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
    class RightHandSide : public Function<dim>
    {
    public:
        RightHandSide () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component = 0) const;

    };

    template <int dim>
    double
    RightHandSide<dim>::value (const Point<dim>   &p,
                               const unsigned int  component) const
    {

        return 0;
    }
    template <int dim>
    void
    RightHandSide<dim>::value_list (const std::vector<Point<dim> > &points,
                                    std::vector<double>            &values,
                                    const unsigned int              component) const
    {
        Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
        for (unsigned int i=0; i<points.size(); ++i)
            values[i] = RightHandSide<dim>::value (points[i], component);
    }
    
    
    template <int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        BoundaryValues () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component = 0) const;
    };
    template <int dim>
    double
    BoundaryValues<dim>::value (const Point<dim>   &p,
                                const unsigned int  component) const
    {

        return p[0]*p[1];
    }
    
    
    template <int dim>
    void
    BoundaryValues<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double>            &values,
                                     const unsigned int              component) const
    {
        Assert (values.size() == points.size(),
                ExcDimensionMismatch (values.size(), points.size()));
        for (unsigned int i=0; i<points.size(); ++i)
            values[i] = BoundaryValues<dim>::value (points[i], component);
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
        
        return  p[1]*p[0];
        
    }
    
   
    
    
    
    template <int dim>
    AdvectionProblem<dim>::AdvectionProblem ()
    :
    dof_handler (triangulation),
    fe(1)
    {}
    
    
    template <int dim>
    AdvectionProblem<dim>::~AdvectionProblem ()
    {
        dof_handler.clear ();
    }
    
    
    template <int dim>
    void AdvectionProblem<dim>::make_grid_and_dofs()
    {
        
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
        triangulation.refine_global (data::global_refinement_level);
        
        
        dof_handler.distribute_dofs (fe);
        hanging_node_constraints.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler,
                                                 hanging_node_constraints);
        hanging_node_constraints.close ();
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp,
                                        hanging_node_constraints,
                                        /*keep_constrained_dofs = */ true);
        sparsity_pattern.copy_from (dsp);
        system_matrix.reinit (sparsity_pattern);
        solution.reinit (dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());
    }
    
    

    
    
    template <int dim>
    void
    AdvectionProblem<dim>::assemble_system ()
    {
        system_matrix = 0;
        
        FullMatrix<double>                   cell_matrix;
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(data::degree+2);
        QGauss<dim-1> face_quadrature_formula(data::degree+2);
        
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);

        
        const AdvectionField<dim> advection_field;
        const BoundaryValues<dim> boundary_values;
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        
        
        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<Tensor<1,dim> > advection_directions (n_q_points);
        std::vector<double>         face_boundary_values (n_face_q_points);
        std::vector<Tensor<1,dim> > face_advection_directions (n_face_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            cell_matrix = 0;
            
        fe_values.reinit (cell);
        advection_field.value_list (fe_values.get_quadrature_points(),
                                    advection_directions);
        
        const double delta = 0.1 * cell->diameter ();
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += ((advection_directions[q_point] *
                                                    fe_values.shape_grad(j,q_point)   *
                                                    (fe_values.shape_value(i,q_point) +
                                                     delta *
                                                     (advection_directions[q_point] *
                                                      fe_values.shape_grad(i,q_point)))) *
                                                   fe_values.JxW(q_point));
            }
        
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            if (cell->face(face)->at_boundary())
            {
                fe_face_values.reinit (cell, face);
                boundary_values.value_list (fe_face_values.get_quadrature_points(),
                                            face_boundary_values);
                advection_field.value_list (fe_face_values.get_quadrature_points(),
                                            face_advection_directions);
                for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    if (fe_face_values.normal_vector(q_point) *
                        face_advection_directions[q_point]
                        < 0)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            for (unsigned int j=0; j<dofs_per_cell; ++j)
                                cell_matrix(i,j) -= (face_advection_directions[q_point] *
                                                               fe_face_values.normal_vector(q_point) *
                                                               fe_face_values.shape_value(i,q_point) *
                                                               fe_face_values.shape_value(j,q_point) *
                                                               fe_face_values.JxW(q_point));
                            
                        }
            }
        
        cell->get_dof_indices (local_dof_indices);
        

        for (unsigned int i=0; i<local_dof_indices.size(); ++i)
        {
            for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));
        }
        }
        
        hanging_node_constraints.condense (system_matrix);
        
    }
    
    
    template <int dim>
    void
    AdvectionProblem<dim>::assemble_rhs ()
    {
        system_rhs=0;
        
        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(data::degree+2);
        QGauss<dim-1> face_quadrature_formula(data::degree+2);
        
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        
        
        const AdvectionField<dim> advection_field;
        const RightHandSide<dim>  right_hand_side;
        const BoundaryValues<dim> boundary_values;
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        
        
        cell_rhs.reinit (dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        
        std::vector<double>         rhs_values (n_q_points);
        std::vector<Tensor<1,dim> > advection_directions (n_q_points);
        std::vector<double>         face_boundary_values (n_face_q_points);
        std::vector<Tensor<1,dim> > face_advection_directions (n_face_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            cell_rhs = 0;
            
            fe_values.reinit (cell);
            advection_field.value_list (fe_values.get_quadrature_points(),
                                        advection_directions);
            right_hand_side.value_list (fe_values.get_quadrature_points(),
                                        rhs_values);
            
            
            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += ((fe_values.shape_value(i,q_point) +
                                     delta *
                                     (advection_directions[q_point] *
                                      fe_values.shape_grad(i,q_point))        ) *
                                    rhs_values[q_point] *
                                    fe_values.JxW (q_point));
                }
            
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary())
                {
                    fe_face_values.reinit (cell, face);
                    boundary_values.value_list (fe_face_values.get_quadrature_points(),
                                                face_boundary_values);
                    advection_field.value_list (fe_face_values.get_quadrature_points(),
                                                face_advection_directions);
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        if (fe_face_values.normal_vector(q_point) *
                            face_advection_directions[q_point]
                            < 0)
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                            {
                                cell_rhs(i) -= (face_advection_directions[q_point] *
                                                fe_face_values.normal_vector(q_point) *
                                                face_boundary_values[q_point]         *
                                                fe_face_values.shape_value(i,q_point) *
                                                fe_face_values.JxW(q_point));
                                
                            }
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }
        }
        
        hanging_node_constraints.condense (system_rhs);
        
    }
    
    
    template <int dim>
    void AdvectionProblem<dim>::solve ()
    {
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult (solution, system_rhs);
        
        
        hanging_node_constraints.distribute (solution);
    }

    template <int dim>
    void AdvectionProblem<dim>::error_analysis ()
    {
        
            ExactSolution<dim> exact_solution;
        
            Vector<double> difference_per_cell (triangulation.n_active_cells());
            
            const QTrapez<1>     q_trapez;
            const QIterated<dim> quadrature (q_trapez, fe.degree+2);
            
            VectorTools::integrate_difference (dof_handler,
                                               solution,
                                               exact_solution,
                                               difference_per_cell,
                                               quadrature,
                                               VectorTools::L2_norm);
            
            const double L2_error = difference_per_cell.l2_norm();
            
            std::cout << "Errors: L2 = " << L2_error << std::endl;
        

    }
    
    

    template <int dim>
    void AdvectionProblem<dim>::run ()
    {

        
            make_grid_and_dofs();
            assemble_system ();
            assemble_rhs ();

            std::cout << "   Number of active cells:       "
            << triangulation.n_active_cells()
            << std::endl;
        
            std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

            solve ();
            error_analysis();
        
            DataOut<dim> data_out;
            data_out.attach_dof_handler (dof_handler);
            data_out.add_data_vector (solution, "solution");
            data_out.build_patches ();
            std::ofstream output ("final-solution.vtk");
            data_out.write_vtk (output);
    }

    
}


int main ()
{
    try
    {
        Step9::MultithreadInfo::set_thread_limit();
        
        
        Step9::AdvectionProblem<2> advection_problem_2d;
        advection_problem_2d.run ();
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
