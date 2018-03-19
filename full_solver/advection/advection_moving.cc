
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
#include <deal.II/base/convergence_table.h>

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
        int degree;
        const double top = 1*1.0;
        const double bottom = 0.0;
        const double right = 1*PI;
        
        const double timestep = 0.001;
        const double final_time = 11*timestep;
        const double error_time = 10;
        
        const double C = 100;
        
        const int global_refinement_level = 3;
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
        void output_results ();
        void output ();
        
        ConvergenceTable     convergence_table;
        
        Triangulation<dim>   triangulation;
        DoFHandler<dim>      dof_handler;
        FE_Q<dim>            fe;
        ConstraintMatrix     hanging_node_constraints;
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> system_matrix;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> nontime_matrix;
        Vector<double>       solution;
        Vector<double>       old_solution;
        Vector<double>       system_rhs;
        Vector<double>       nontime_rhs;
        
        double               time;
        double               time_step;
        unsigned int         timestep_number;
        
    };
    
    
    template <int dim>
    class Advection : public Function<dim>
    {
    public:
        Advection () : Function<dim>(dim) {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    
    template <int dim>
    void
    Advection<dim>::vector_value (const Point<dim> &p,
                                          Vector<double>   &values) const
    {
        values(0) = 0;
        values(1) = -p[1]*p[1];
    }

    
    
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
        value[0] = 0;
        value[1] = -p[1]*p[1];
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
//        const double time = this->get_time();
        return data::C*p[1];
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
        const double time = this->get_time();
        return data::C*time*p[1];
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
        const double time = this->get_time();
        return  data::C*p[1]*time;
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
        return 0;
    }
    
    // Use this if you know where the inflow boundaries are and can define them
    template <int dim>
    class DiriFunction : public Function<dim>
    {
    public:
        DiriFunction () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double DiriFunction<dim>::value (const Point<dim>  &p,
                                        const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        return  data::C*p[1]*time;
    }
    
    template <int dim>
    AdvectionProblem<dim>::AdvectionProblem ()
    :
    dof_handler (triangulation),
    fe(data::degree)
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
        typename Triangulation<dim>::cell_iterator
        cell = triangulation.begin (),
        endc = triangulation.end();
        for (; cell!=endc; ++cell)
            for (unsigned int face_number=0;
                 face_number<GeometryInfo<dim>::faces_per_cell;
                 ++face_number)
                if (std::fabs(cell->face(face_number)->center()(1) - (data::top)) < 1e-12)
                    cell->face(face_number)->set_boundary_id (1);
        
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
        mass_matrix.reinit (sparsity_pattern);
        nontime_matrix.reinit (sparsity_pattern);
        solution.reinit (dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());
        nontime_rhs.reinit (dof_handler.n_dofs());
    }
    
    template <int dim>
    void
    AdvectionProblem<dim>::assemble_system ()
    {
        system_matrix = 0;
        nontime_matrix = 0;
        
        MatrixCreator::create_mass_matrix(dof_handler,
                                          QGauss<dim>(data::degree+4),
                                          mass_matrix);
        
        FullMatrix<double>                   cell_matrix;
        
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(data::degree+3);
        QGauss<dim-1> face_quadrature_formula(data::degree+3);
        
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        
        const AdvectionField<dim> advection_field;
        
        const unsigned int dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int n_q_points      = fe_values.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values.get_quadrature().size();
        
        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<Tensor<1,dim> > advection_directions (n_q_points);
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
            
//            const double delta = 0.1 * cell->diameter ();

            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    // (v_r.grad phi, psi+d*v_r.grad_psi)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += 0*((advection_directions[q_point] *
                                              fe_values.shape_grad(j,q_point)   *
                                              (fe_values.shape_value(i,q_point)
//                                               +
//                                               delta *
//                                               (advection_directions[q_point] *
//                                                fe_values.shape_grad(i,q_point))
                                               )) *
                                             fe_values.JxW(q_point));
                    
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                    nontime_matrix.add (local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i,j));
            }
        }
    }

    template <int dim>
    void
    AdvectionProblem<dim>::assemble_rhs ()
    {
        system_rhs=0;
        nontime_rhs=0;
        
        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(data::degree+4);
        QGauss<dim-1> face_quadrature_formula(data::degree+4);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        
        const AdvectionField<dim> advection_field;
        
        RightHandSide<dim>  right_hand_side;
        right_hand_side.set_time(time);
        
        BoundaryValues<dim> boundary_values;
        boundary_values.set_time(time);
        
        
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
            
//            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += ((fe_values.shape_value(i,q_point)
//                                     +
//                                     delta *
//                                     (advection_directions[q_point] *
//                                      fe_values.shape_grad(i,q_point))
                                     ) *
                                    rhs_values[q_point] *
                                    fe_values.JxW (q_point));
//                    std::cout << component_i << std::endl;
                    
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                nontime_rhs(local_dof_indices[i]) += cell_rhs(i);
                
            }
        }
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
        exact_solution.set_time(time);
        
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
        
        convergence_table.add_value("degree", data::degree);
        convergence_table.add_value("refinement level", data::global_refinement_level);
        convergence_table.add_value("L2", L2_error);
        
        convergence_table.set_precision("L2", 4);
        convergence_table.set_scientific("L2", true);
        
        convergence_table.set_tex_caption("degree", "\\# degree");
        convergence_table.set_tex_caption("refinement level", "\\# refinement level");
        convergence_table.set_tex_caption("L2", "L^2-error");
        
        std::string error_filename = "error";
        error_filename += ".tex";
        std::ofstream error_table_file(error_filename.c_str());
        convergence_table.write_tex(error_table_file);
        
        std::cout << std::endl;
        convergence_table.write_text(std::cout);
    }
    
    template <int dim>
    void AdvectionProblem<dim>::output_results ()
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
    void AdvectionProblem<dim>::run ()
    {
        
        timestep_number = 0;
        time            = 0;
        
        make_grid_and_dofs();
        assemble_system ();
        assemble_rhs ();
//
//        SparseDirectUMFPACK  A_direct;
//        A_direct.initialize(nontime_matrix);
//        A_direct.vmult (solution, nontime_rhs);
//
//        hanging_node_constraints.distribute (solution);
//        output_results();
//        error_analysis();
        
        
        VectorTools::interpolate(dof_handler,
                                 InitialFunction<dim>(),
                                 old_solution);
        
        solution = old_solution;
        output_results();
        time_step = data::timestep;

        while (time <= data::final_time)
        {
            time += time_step;
            ++timestep_number;

            assemble_rhs ();

            mass_matrix.vmult(system_rhs, old_solution);
            system_rhs.add(time_step,nontime_rhs);

            system_matrix.copy_from(mass_matrix);
            system_matrix.add(time_step,nontime_matrix);
            
            
            std::map<types::global_dof_index,double> boundary_values;
            DiriFunction<dim> diri_values;
            diri_values.set_time(time);
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      1,
                                                      diri_values,
                                                      boundary_values);
            MatrixTools::apply_boundary_values (boundary_values,
                                                system_matrix,
                                                solution,
                                                system_rhs);
            
            hanging_node_constraints.condense (system_rhs);
            hanging_node_constraints.condense (system_matrix);

            solve ();
            output_results ();
            // error_analysis();

            if (timestep_number == data::error_time)
            {

                std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

                std::cout << "   Number of active cells:       "
                << triangulation.n_active_cells()
                << std::endl;

                std::cout << "   Number of degrees of freedom: "
                << dof_handler.n_dofs()
                << std::endl;

                error_analysis();
            }

            old_solution = solution;
        }
        
        std::cout << " Finite element degree & refinement level:            "
        << fe.degree << " and " << data::global_refinement_level
        << std::endl;
        
    }
}


int main ()
{
    try
    {
        Step9::MultithreadInfo::set_thread_limit();
        
        for (Step9::data::degree = 1; Step9::data::degree<=1; Step9::data::degree++)
        {
            Step9::AdvectionProblem<Step9::data::dimension> advection_problem_2d;
            advection_problem_2d.run ();
        }
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

