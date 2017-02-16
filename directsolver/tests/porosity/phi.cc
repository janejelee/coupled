
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
        
        const double timestep = 0.01;
        const double final_time = 15*timestep;
        const double error_time = 13;
        
        
    }
    
   

    
    template<int dim>
    class PorosityEquation
    {
    public:
        PorosityEquation();
        void run();
        
    private:
        void make_grid_and_dofs ();
        void setup_system();
        void assemble_rhs ();
        void solve_time_step();
        void output_results() const;
        void refine_mesh (const unsigned int min_grid_level,
                          const unsigned int max_grid_level);
        
        Triangulation<dim>   triangulation;
        FE_Q<dim>            fe;
        DoFHandler<dim>      dof_handler;
        
        ConstraintMatrix     constraints;
        
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> mass_matrix;
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
        
        return 0.0;
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
        
        return  time;
        
    }
    
    
    
    
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

            return 1;
    }
    
    
    
    template<int dim>
    class BoundaryValues : public Function<dim>
    {
    public:
        virtual double value (const Point<dim>  &p,
                              const unsigned int component = 0) const;
    };
    
    
    
    template<int dim>
    double BoundaryValues<dim>::value (const Point<dim> &p,
                                       const unsigned int component) const
    {
        Assert(component == 0, ExcInternalError());
        const double time = this->get_time();
        
        return time;
    }
    
    
    
    template <int dim>
    class AdvectionField : public TensorFunction<1,dim,double>
    {
    public:
        AdvectionField () : TensorFunction<1,dim, double> ()
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
        value[1] = 0.0;
        
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
    

    
    
    
    template<int dim>
    PorosityEquation<dim>::PorosityEquation ()
    :
    fe(1),
    dof_handler(triangulation),
    time_step(data::timestep),
    theta(0.5)
    {}
    
    
    template <int dim>
    void PorosityEquation<dim>::make_grid_and_dofs ()
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
        /*
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] == data::top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] == data::bottom)
                    cell->face(f)->set_all_boundary_ids(2);
         */
        
        triangulation.refine_global (initial_global_refinement+2);

        
        dof_handler.distribute_dofs(fe);
        
        std::cout << std::endl
        << "==========================================="
        << std::endl
        << "Number of active cells: " << triangulation.n_active_cells()
        << std::endl
        << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl
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
        
        mass_matrix.reinit(sparsity_pattern);
        advection_matrix.reinit(sparsity_pattern);
        system_matrix.reinit(sparsity_pattern);
        
        solution.reinit(dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
        system_rhs_mass.reinit(dof_handler.n_dofs());
        
        
    }
    
    
    template<int dim>
    void PorosityEquation<dim>::setup_system()
    {
        
        system_matrix = 0;
        system_rhs = 0;
        mass_matrix = 0;
        
        MatrixCreator::create_mass_matrix(dof_handler,
                                          QGauss<dim>(fe.degree+1),
                                          mass_matrix);
   
        QGauss<dim>  quadrature_formula(data::degree+2);
        QGauss<dim-1> face_quadrature_formula(data::degree+2);
        
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        
        
        const unsigned int   dofs_per_cell = fe.dofs_per_cell;
        const unsigned int   n_q_points    = quadrature_formula.size();
        
        
        const AdvectionField<dim> advection_field;
        std::vector<Tensor<1,dim> > advection_directions (n_q_points);
        
        advection_field.value_list (fe_values.get_quadrature_points(),
                                    advection_directions);
        
        
        
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
                        cell_advection_matrix(i,j) +=
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
                {
                    advection_matrix.add (local_dof_indices[i],
                                          local_dof_indices[j],
                                          cell_advection_matrix(i,j));
                }
                
            }
        }
        
    }
    
    template <int dim>
    void PorosityEquation<dim>::assemble_rhs ()
    {
        Vector<double> tmp;
        tmp.reinit (solution.size());
        
        mass_matrix.vmult(system_rhs, old_solution);
        
        advection_matrix.vmult(tmp, old_solution);
        system_rhs.add(-time_step, tmp);
        
        
        // extra right hand side stuff
        RightHandSide<dim> rhs_function;
        
        rhs_function.set_time(time - time_step);
        VectorTools::create_right_hand_side(dof_handler,
                                            QGauss<dim>(fe.degree+1),
                                            rhs_function,
                                            tmp);
        
        
        system_rhs.add(time_step, tmp);
        


        /*
        QGauss<dim>   quadrature_formula(degree+2);
        
        FEValues<dim> fe_values (fe, quadrature_formula,
                                 update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);

        
        const unsigned int   dofs_per_cell   = fe.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<Vector<double> > old_solution_values(n_q_points, Vector<double>);
        std::vector<Vector<double> > present_solution_values(n_q_points, Vector<double>);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        BoundaryValues<dim> boundary_values;
        
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            local_rhs = 0;
            fe_values.reinit (cell);
            fe_values.get_function_values (old_solution, old_solution_values);
            fe_values.get_function_values (solution, present_solution_values);
            
            for (unsigned int q=0; q<n_q_points; ++q)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    const double old_s = old_solution_values[q](dim+1);
                    Tensor<1,dim> present_u;
                    
                    for (unsigned int d=0; d<dim; ++d)
                        present_u[d] = present_solution_values[q](d);
                    const double        phi_i_s      = fe_values[saturation].value (i, q);
                    const Tensor<1,dim> grad_phi_i_s = fe_values[saturation].gradient (i, q);
                    local_rhs(i) += (time_step *
                                     fractional_flow(old_s,viscosity) *
                                     present_u *
                                     grad_phi_i_s
                                     +
                                     old_s * phi_i_s)
                    *
                    fe_values.JxW(q);
                }
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                system_rhs(local_dof_indices[i]) += local_rhs(i);
        }
         */
    }
    
    
    template<int dim>
    void PorosityEquation<dim>::solve_time_step()
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
    void PorosityEquation<dim>::output_results() const
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
    
    
 
    
    
    template<int dim>
    void PorosityEquation<dim>::run()
    {
        make_grid_and_dofs();
        
        setup_system();
        
        unsigned int pre_refinement_step = 0;
        
        Vector<double> tmp;
        tmp.reinit (solution.size());
        
        
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
            
            std::cout << "Time step " << timestep_number << " at t=" << time
            << std::endl;
            
            
            assemble_rhs ();
            
            system_matrix.copy_from(mass_matrix);
            
            constraints.condense (system_matrix, system_rhs);
            
            {
                BoundaryValues<dim> boundary_values_function;
                boundary_values_function.set_time(time);
                
                std::map<types::global_dof_index, double> boundary_values;

                VectorTools::interpolate_boundary_values(dof_handler,
                                                         0,
                                                         boundary_values_function,
                                                         boundary_values);
                
                
                MatrixTools::apply_boundary_values(boundary_values,
                                                   system_matrix,
                                                   solution,
                                                   system_rhs);
            }
            
            solve_time_step();
            
            output_results();
            
            {
                ExactSolution<dim> exact_solution;
                exact_solution.set_time(time);
                Vector<double> difference_per_cell (triangulation.n_active_cells());
                std::cout << triangulation.n_active_cells() << std::endl;
                
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
        
        PorosityEquation<2> porosity_solver;
        porosity_solver.run();
        
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
