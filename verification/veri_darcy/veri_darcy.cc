#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/sparse_direct.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace FullSolver
{
    using namespace dealii;
    using namespace numbers;
    
    namespace data
    {
        const unsigned int base_degree = 2;
        const unsigned int degree_vr   = base_degree +1;
        const unsigned int degree_pr   = base_degree;
        const unsigned int degree_pf   = base_degree;
        const unsigned int degree_vf   = base_degree +1;
        const unsigned int degree_phi  = base_degree;
        const unsigned int degree_T    = base_degree;
        
        
        const int refinement_level = 3;
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        const double patm = 1.0;
        const double rho_f = 1.0;
        const double rho_r = 2.71;
        const double phi0 = 0.7;
        const double lambda = 1;
        const double k = 1;
        const double A = 0.1;
        
        
        ConvergenceTable                        convergence_table;
        ConvergenceTable                        convergence_table_rate;
    }
    
    using namespace data;
    
    template <int dim>
    class FullMovingMesh
    {
    public:
        FullMovingMesh (const unsigned int degree_vr, const unsigned int degree_pr,
                        const unsigned int degree_pf, const unsigned int degree_vf,
                        const unsigned int degree_phi, const unsigned int degree_T);
        void run ();
    private:
        void make_initial_grid ();
        void setup_dofs_rock ();
        void setup_dofs_fluid ();
        void setup_dofs_phi ();
        void setup_dofs_T ();
        void apply_BC_rock ();
        void assemble_system_rock ();
        void assemble_system_pf ();
        void assemble_system_vf ();
        void assemble_system_phi ();
        void assemble_system_T ();
        void assemble_rhs_phi ();
        void assemble_rhs_T ();
        void solve_rock ();
        void solve_fluid ();
        void solve_phi ();
        void solve_T ();
        void compute_errors ();
        void output_results () const;
        void move_mesh ();
        void print_mesh ();
        
        const unsigned int        degree_vr;
        const unsigned int        degree_pr;
        Triangulation<dim>        triangulation;
        FESystem<dim>             fe_rock;
        DoFHandler<dim>           dof_handler_rock;
        ConstraintMatrix          constraints_rock;
        BlockSparsityPattern      sparsity_pattern_rock;
        BlockSparseMatrix<double> system_matrix_rock;
        BlockVector<double>       solution_rock;
        BlockVector<double>       solution_rock_nonmag;
        BlockVector<double>       system_rhs_rock;
        
        const unsigned int        degree_pf;
        FESystem<dim>             fe_pf;
        DoFHandler<dim>           dof_handler_pf;
        ConstraintMatrix          constraints_pf;
        BlockSparsityPattern      sparsity_pattern_pf;
        BlockSparseMatrix<double> system_matrix_pf;
        BlockVector<double>       solution_pf;
        BlockVector<double>       system_rhs_pf;
        
        const unsigned int        degree_vf;
        FESystem<dim>             fe_vf;
        DoFHandler<dim>           dof_handler_vf;
        ConstraintMatrix          constraints_vf;
        SparsityPattern              sparsity_pattern_vf;
        SparseMatrix<double>      system_matrix_vf;
        SparseMatrix<double>       mass_matrix_vf;
        Vector<double>              solution_vf;
        Vector<double>              system_rhs_vf;
        
        DoFHandler<dim>           dof_handler_phi;
        FE_Q<dim>                 fe_phi;
        ConstraintMatrix          hanging_node_constraints_phi;
        SparsityPattern           sparsity_pattern_phi;
        SparseMatrix<double>      system_matrix_phi;
        SparseMatrix<double>      mass_matrix_phi;
        SparseMatrix<double>      nontime_matrix_phi;
        Vector<double>            solution_phi;
        Vector<double>            old_solution_phi;
        Vector<double>            system_rhs_phi;
        Vector<double>            nontime_rhs_phi;
        
        DoFHandler<dim>           dof_handler_T;
        FE_Q<dim>                 fe_T;
        ConstraintMatrix          hanging_node_constraints_T;
        SparsityPattern           sparsity_pattern_T;
        SparseMatrix<double>      system_matrix_T;
        SparseMatrix<double>      mass_matrix_T;
        SparseMatrix<double>      mass_matrix_T_old;
        SparseMatrix<double>      nontime_matrix_T;
        Vector<double>            solution_T;
        Vector<double>            old_solution_T;
        Vector<double>            system_rhs_T;
        Vector<double>            nontime_rhs_T;
        
        Vector<double>            displacement;
        Vector<double>            initial_pf;
        
        double time;
        double timestep;
        unsigned int timestep_number;
        
    };

    template <int dim>
    class InitialFunction_rock : public Function<dim>
    {
    public: InitialFunction_rock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double> &value) const;
    };
    
    template <int dim>
    class ExtraRHSpf : public Function<dim>
    {
    public: ExtraRHSpf () : Function<dim>() {}
        virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    };
    
    template <int dim>
    class DirichletFunction : public Function<dim>
    {
    public: DirichletFunction () : Function<dim>() {}
        virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    };
    
    template <int dim>
    class InitialFunction_phi : public Function<dim>
    {
    public: InitialFunction_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    };
    
    template <int dim>
    class FluxFunction : public Function<dim>
    {
    public: FluxFunction () : Function<dim>(dim) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactFunction_pf : public Function<dim>
    {
    public: ExactFunction_pf () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double> &value) const;
    };
    
    template <int dim>
    class ExactSolution_vf : public Function<dim>
    {
    public: ExactSolution_vf () : Function<dim>(dim) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void InitialFunction_rock<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
        values(0) = 0.0;
        values(1) = -p[1]*p[1];
        values(2) = -p[0]*(p[1]-1.0/3.0*p[1]*p[1]*p[1]); //this is actually pf. dont need pr in veri
    }
    
    template <int dim>
    void ExactFunction_pf<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
        values(0) = (p[1]-1./3*p[1]*p[1]*p[1]); //ux = -grad pf x
        values(1) = p[0]*(1-p[1]*p[1]);  // uy = -grad pf y
        values(2) = -p[0]*(p[1]-1.0/3.0*p[1]*p[1]*p[1]); //p
    }
    
    template <int dim>
    void FluxFunction<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        values(0) = (p[1]-1.0/3.0*p[1]*p[1]*p[1]);
        values(1) = p[0]*(1-p[1]*p[1]);
    }
    
    template <int dim>
    void ExactSolution_vf<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        const double time = this->get_time();
        const double phi = phi0 + A*phi0*time*p[1];
        
        values(0) = lambda * k / phi *(p[1]-1.0/3.0*p[1]*p[1]*p[1]) ;
        values(1) = -p[1]*p[1] + lambda * k / phi * ( p[0]*(1-p[1]*p[1]) - rho_f);
    }
    
    template <int dim>
    double ExtraRHSpf<dim>::value (const Point<dim>  &p, const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        const double phi = phi0 + A*phi0*time*p[1];
        
        return 2*lambda*k*p[0]*p[1] + 2*p[1];
    }
    
    
    template <int dim>
    double DirichletFunction<dim>::value (const Point<dim>  &p, const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        const double phi = phi0 + A*phi0*time*p[1];
        
        return -p[0]*(p[1]-1.0/3.0*p[1]*p[1]*p[1]);
    }
    
    
    template <int dim>
    double InitialFunction_phi<dim>::value (const Point<dim>  &p, const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        return phi0 + A*phi0*time*p[1];
    }

    
    template <int dim>
    void unitz (const std::vector<Point<dim> > &points,
                std::vector<Tensor<1, dim> >   &values)
    {
        for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
        {
            values[point_n][0] = 0.0;
            values[point_n][1] = 1.0;
        }
    }
    
    template <int dim>
    FullMovingMesh<dim>::FullMovingMesh (const unsigned int degree_vr, const unsigned int degree_pr,
                                         const unsigned int degree_pf, const unsigned int degree_vf,
                                         const unsigned int degree_phi, const unsigned int degree_T)
    :
    degree_vr (degree_vr),
    degree_pr (degree_pr),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe_rock (FE_Q<dim>(degree_vr), dim,
             FE_Q<dim>(degree_pr), 1),
    dof_handler_rock (triangulation),
    
    degree_pf (degree_pf),
    fe_pf (FE_RaviartThomas<dim>(degree_pf), 1,
           FE_DGQ<dim>(degree_pf), 1),
    dof_handler_pf (triangulation),
    
    degree_vf (degree_vf),
    fe_vf (FE_Q<dim>(degree_vf), dim),
    dof_handler_vf (triangulation),
    
    dof_handler_phi (triangulation),
    fe_phi (degree_phi),
    
    dof_handler_T (triangulation),
    fe_T (degree_T)
    
    {}

    template <int dim>
    void FullMovingMesh<dim>::make_initial_grid ()
    {
        std::vector<unsigned int> subdivisions (dim, 1);
        subdivisions[0] = 5;
        
        const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>( left, bottom) :
                                        Point<dim>(-2,0,-1));
        const Point<dim> top_right   = (dim == 2 ?
                                        Point<dim>( right, top) :
                                        Point<dim>(0,1,0));
        
        GridGenerator::subdivided_hyper_rectangle (triangulation, subdivisions, bottom_left, top_right);
        
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] ==  top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] ==  bottom)
                    cell->face(f)->set_all_boundary_ids(2);
        
        triangulation.refine_global (refinement_level+1);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_rock ()
    {
        system_matrix_rock.clear ();
        dof_handler_rock.distribute_dofs (fe_rock);
        DoFRenumbering::Cuthill_McKee (dof_handler_rock);
        std::vector<unsigned int> block_component (dim+1,0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise (dof_handler_rock, block_component);
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler_rock, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
        n_p = dofs_per_block[1];
        std::cout
        << "   Refinement level: "
        << refinement_level
        << std::endl
        << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom in rock problem: "
        << dof_handler_rock.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
        BlockDynamicSparsityPattern dsp_rock (2,2);
        dsp_rock.block(0,0).reinit (n_u, n_u);
        dsp_rock.block(1,0).reinit (n_p, n_u);
        dsp_rock.block(0,1).reinit (n_u, n_p);
        dsp_rock.block(1,1).reinit (n_p, n_p);
        dsp_rock.collect_sizes();
        DoFTools::make_sparsity_pattern (dof_handler_rock, dsp_rock, constraints_rock, false);
        sparsity_pattern_rock.copy_from (dsp_rock);
        
        system_matrix_rock.reinit (sparsity_pattern_rock);
        solution_rock.reinit (2);
        solution_rock.block(0).reinit (n_u);
        solution_rock.block(1).reinit (n_p);
        solution_rock.collect_sizes ();
        solution_rock_nonmag.reinit (2);
        solution_rock_nonmag.block(0).reinit (n_u);
        solution_rock_nonmag.block(1).reinit (n_p);
        solution_rock_nonmag.collect_sizes ();
        system_rhs_rock.reinit (2);
        system_rhs_rock.block(0).reinit (n_u);
        system_rhs_rock.block(1).reinit (n_p);
        system_rhs_rock.collect_sizes ();
        displacement.reinit(n_u);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_fluid ()
    {
        dof_handler_pf.distribute_dofs (fe_pf);
        dof_handler_vf.distribute_dofs (fe_vf);
        
        DoFRenumbering::component_wise (dof_handler_pf);
        std::vector<types::global_dof_index> dofs_per_component (dim+1);
        DoFTools::count_dofs_per_component (dof_handler_pf, dofs_per_component);
        const unsigned int n_u = dofs_per_component[0],
        n_pf = dofs_per_component[dim];
        
        DoFRenumbering::Cuthill_McKee (dof_handler_vf);
        std::vector<unsigned int> block_component (dim,0);
        DoFRenumbering::component_wise (dof_handler_vf, block_component);
        
        std::vector<types::global_dof_index> dofs_per_block (1);
        DoFTools::count_dofs_per_block (dof_handler_vf, dofs_per_block, block_component);
        const unsigned int n_vf = dofs_per_block[0];
        
        std::cout << "   Number of degrees of freedom in fluid problem: "
        << dof_handler_pf.n_dofs()
        << " (" << n_u << '+' << n_pf << '+' << n_vf << ')'
        << std::endl;
        
        constraints_pf.clear ();
        
        DoFTools::make_hanging_node_constraints (dof_handler_pf, constraints_pf);
        
        VectorTools::project_boundary_values_div_conforming
                    (dof_handler_pf, 0, FluxFunction<dim>(), 2, constraints_pf);

        VectorTools::project_boundary_values_div_conforming
                    (dof_handler_pf, 0, FluxFunction<dim>(), 0, constraints_pf);
        
        constraints_pf.close ();
        
        BlockDynamicSparsityPattern dsp_pf(2, 2);
        dsp_pf.block(0, 0).reinit (n_u, n_u);
        dsp_pf.block(1, 0).reinit (n_pf, n_u);
        dsp_pf.block(0, 1).reinit (n_u, n_pf);
        dsp_pf.block(1, 1).reinit (n_pf, n_pf);
        
        dsp_pf.collect_sizes ();
        DoFTools::make_sparsity_pattern (dof_handler_pf, dsp_pf, constraints_pf, true);
        
        sparsity_pattern_pf.copy_from(dsp_pf);
        system_matrix_pf.reinit (sparsity_pattern_pf);
        solution_pf.reinit (2);
        solution_pf.block(0).reinit (n_u);
        solution_pf.block(1).reinit (n_pf);
        solution_pf.collect_sizes ();
        
        system_rhs_pf.reinit (2);
        system_rhs_pf.block(0).reinit (n_u);
        system_rhs_pf.block(1).reinit (n_pf);
        system_rhs_pf.collect_sizes ();
        
        system_matrix_vf.clear ();
        constraints_vf.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_vf, constraints_vf);
        constraints_vf.close();
        DynamicSparsityPattern dsp_vf(dof_handler_vf.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_vf, dsp_vf, constraints_vf,
                                        /*keep_constrained_dofs = */ true);
        sparsity_pattern_vf.copy_from(dsp_vf);
        mass_matrix_vf.reinit (sparsity_pattern_vf);
        system_matrix_vf.reinit (sparsity_pattern_vf);
        solution_vf.reinit (dof_handler_vf.n_dofs());
        system_rhs_vf.reinit (dof_handler_vf.n_dofs());
        
        MatrixCreator::create_mass_matrix(dof_handler_vf, QGauss<dim>(degree_vf+2),
                                          system_matrix_vf);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_phi ()
    {
        dof_handler_phi.distribute_dofs (fe_phi);
        hanging_node_constraints_phi.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_phi, hanging_node_constraints_phi);
        hanging_node_constraints_phi.close ();
        std::cout << "   Number of degrees of freedom in porosity problem: "
        << dof_handler_phi.n_dofs()
        << std::endl;
        DynamicSparsityPattern dsp_phi(dof_handler_phi.n_dofs(), dof_handler_phi.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_phi, dsp_phi, hanging_node_constraints_phi,
                                        /*keep_constrained_dofs = */ true);
        sparsity_pattern_phi.copy_from (dsp_phi);
        system_matrix_phi.reinit (sparsity_pattern_phi);
        mass_matrix_phi.reinit (sparsity_pattern_phi);
        nontime_matrix_phi.reinit (sparsity_pattern_phi);
        solution_phi.reinit (dof_handler_phi.n_dofs());
        old_solution_phi.reinit(dof_handler_phi.n_dofs());
        system_rhs_phi.reinit (dof_handler_phi.n_dofs());
        nontime_rhs_phi.reinit (dof_handler_phi.n_dofs());
        initial_pf.reinit (dof_handler_phi.n_dofs());
        
        MatrixCreator::create_mass_matrix(dof_handler_phi, QGauss<dim>(degree_phi+2),
                                          mass_matrix_phi);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_T ()
    {
        dof_handler_T.distribute_dofs (fe_T);
        DoFRenumbering::Cuthill_McKee (dof_handler_T);
        
        hanging_node_constraints_T.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_T, hanging_node_constraints_T);
        hanging_node_constraints_T.close ();
        std::cout << "   Number of degrees of freedom in temperature problem: "
        << dof_handler_T.n_dofs()
        << std::endl;
        DynamicSparsityPattern dsp_T (dof_handler_T.n_dofs(), dof_handler_T.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler_T, dsp_T);
        hanging_node_constraints_T.condense (dsp_T);
        sparsity_pattern_T.copy_from (dsp_T);
        
        system_matrix_T.reinit (sparsity_pattern_T);
        mass_matrix_T.reinit (sparsity_pattern_T);
        mass_matrix_T_old.reinit (sparsity_pattern_T);
        nontime_matrix_T.reinit (sparsity_pattern_T);
        solution_T.reinit (dof_handler_T.n_dofs());
        old_solution_T.reinit(dof_handler_T.n_dofs());
        system_rhs_T.reinit (dof_handler_T.n_dofs());
        nontime_rhs_T.reinit (dof_handler_T.n_dofs());
    }
 
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_pf ()
    {
        QGauss<dim>   quadrature_formula(degree_pf+2);
        QGauss<dim-1> face_quadrature_formula(degree_pf+2);
        
        FEValues<dim> fe_values_pf (fe_pf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_pf (fe_pf, face_quadrature_formula,
                                             update_values    | update_normal_vectors |
                                             update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                      update_values    | update_gradients |
                                      update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_rock (fe_rock, face_quadrature_formula,
                                               update_values    | update_normal_vectors |
                                               update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    | update_gradients |
                                     update_quadrature_points  | update_JxW_values);
        
        const unsigned int   dofs_per_cell   = fe_pf.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        std::vector<double> div_vr_values (n_q_points);
        std::vector<double> rhs_values (n_q_points);
        std::vector<double> pr_boundary_values (n_face_q_points);
        std::vector<double> phi_values (n_q_points);
        std::vector<Tensor<1,dim>>     unitz_values (n_q_points);
        std::vector<Tensor<1,dim>>     grad_phi_values (n_q_points);
        
        ExtraRHSpf<dim> rhs_function;
        rhs_function.set_time (time);
        
        DirichletFunction<dim> boundary_function;
        boundary_function.set_time (time);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_pf.begin_active(),
        endc = dof_handler_pf.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        phi_cell = dof_handler_phi.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell, ++phi_cell)
        {
            fe_values_pf.reinit (cell);
            fe_values_rock.reinit(vr_cell);
            fe_values_phi.reinit(phi_cell);
            
            local_matrix = 0;
            local_rhs = 0;
            
            unitz (fe_values_pf.get_quadrature_points(), unitz_values);
            fe_values_phi.get_function_values (solution_phi, phi_values);
            fe_values_phi.get_function_gradients (solution_phi, grad_phi_values);
            fe_values_rock[velocities].get_function_divergences (solution_rock, div_vr_values);
            rhs_function.value_list (fe_values_pf.get_quadrature_points(), rhs_values);
            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    const Tensor<1,dim> phi_i_u     = fe_values_pf[velocities].value (i, q);
                    const double        div_phi_i_u = fe_values_pf[velocities].divergence (i, q);
                    const double        phi_i_p     = fe_values_pf[pressure].value (i, q);
                    
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        const Tensor<1,dim> phi_j_u     = fe_values_pf[velocities].value (j, q);
                        const double        div_phi_j_u = fe_values_pf[velocities].divergence (j, q);
                        const double        phi_j_p     = fe_values_pf[pressure].value (j, q);
                        
                        local_matrix(i,j) += ( phi_i_u * phi_j_u
                                              - div_phi_i_u * phi_j_p
                                              - lambda * k * phi_i_p * div_phi_j_u)
                                                    * fe_values_pf.JxW(q);
                    }
                    
                    local_rhs(i) += phi_i_p *
                                    (div_vr_values[q] + rhs_values[q])
                                            * fe_values_pf.JxW(q);
                }
            }
            
            for (unsigned int face_no=0;
                 face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
                if (cell->face(face_no)->at_boundary()
                    &&
                    (cell->face(face_no)->boundary_id() == 1)) // Basin top has boundary id 1
                {
                    fe_face_values_pf.reinit (cell, face_no);
                    fe_face_values_rock.reinit (vr_cell, face_no);
                    
                    fe_face_values_rock[pressure].get_function_values
                                                    (solution_rock, pr_boundary_values);
//                    boundary_function.value_list (fe_face_values_pf.get_quadrature_points(), pr_boundary_values);

                    
                    // DIRICHLET CONDITION FOR TOP. pf = pr at top of basin
                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            local_rhs(i) += ( - pr_boundary_values[q] *
                                              fe_face_values_pf[velocities].value (i, q) *
                                              fe_face_values_pf.normal_vector(q) *
                                              fe_face_values_pf.JxW(q));
                        }
                }

            cell->get_dof_indices (local_dof_indices);
            constraints_pf.distribute_local_to_global (local_matrix, local_rhs, local_dof_indices,
                                                         system_matrix_pf, system_rhs_pf);
        }

    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_vf ()
    {
        system_rhs_vf=0;
        
        QGauss<dim>   quadrature_formula(degree_vf+2);
        FEValues<dim> fe_values_vf (fe_vf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_pf (fe_pf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                      update_values    | update_gradients |
                                      update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    | update_gradients |
                                     update_quadrature_points  | update_JxW_values);
        
        const unsigned int   dofs_per_cell   = fe_vf.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        std::vector<Tensor<1,dim>>     u_values (n_q_points);
        std::vector<Tensor<1,dim>>     unitz_values (n_q_points);
        std::vector<Tensor<1,dim>>     vr_values (n_q_points);
        std::vector<double>            pf_values (n_q_points);
        std::vector<double>            phi_values (n_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_vf.begin_active(),
        endc = dof_handler_vf.end();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = dof_handler_pf.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        phi_cell = dof_handler_phi.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++pf_cell, ++vr_cell, ++phi_cell)
        {
            fe_values_vf.reinit(cell);
            fe_values_pf.reinit(pf_cell);
            fe_values_rock.reinit(vr_cell);
            fe_values_phi.reinit(phi_cell);
            
            local_rhs = 0;
            
            unitz (fe_values_vf.get_quadrature_points(), unitz_values);
            fe_values_pf[velocities].get_function_values (solution_pf, u_values);
            fe_values_rock[velocities].get_function_values (solution_rock, vr_values);
            fe_values_pf[pressure].get_function_values (solution_pf, pf_values);
            fe_values_phi.get_function_values (solution_phi, phi_values);
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                const unsigned int
                component_i = fe_vf.system_to_component_index(i).first;
                
                for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                    local_rhs(i) +=    ( vr_values[q_point][component_i]
                                        + lambda*k/phi_values[q_point] *
                                        (u_values[q_point][component_i] // - grad pf
                                         - rho_f*unitz_values[q_point][component_i])
                                        ) * fe_values_vf.shape_value(i,q_point) *
                                        fe_values_vf.JxW(q_point);
                }
            }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                system_rhs_vf(local_dof_indices[i]) += local_rhs(i);
            }
        }
    }
   
    
    template <int dim>
    void FullMovingMesh<dim>::solve_fluid ()
    {
        std::cout << "   Solving for Darcy system..." << std::endl;
        
        SparseDirectUMFPACK  pf_direct;
        pf_direct.initialize(system_matrix_pf);
        pf_direct.vmult (solution_pf, system_rhs_pf);
        constraints_pf.distribute (solution_pf);
        
        assemble_system_vf ();
        std::cout << "   Solving for v_f..." << std::endl;
        SparseDirectUMFPACK  vf_direct;
        vf_direct.initialize(system_matrix_vf);
        vf_direct.vmult (solution_vf, system_rhs_vf);
        constraints_vf.distribute (solution_vf);

    }
    
    template <int dim>
    void FullMovingMesh<dim>::output_results ()  const
    {
        //FLUID
        std::vector<std::string> pf_names (dim, "uf");
        pf_names.push_back ("p_f");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        pf_component_interpretation
        (dim+1, DataComponentInterpretation::component_is_scalar);
        for (unsigned int i=0; i<dim; ++i)
            pf_component_interpretation[i]
            = DataComponentInterpretation::component_is_part_of_vector;
        DataOut<dim> data_out_fluid;
        data_out_fluid.add_data_vector (dof_handler_pf, solution_pf,
                                        pf_names, pf_component_interpretation);
        data_out_fluid.add_data_vector (dof_handler_vf, solution_vf, "v_f");
        
        data_out_fluid.build_patches ();
        const std::string filename_fluid = "solution_fluid-"
        + Utilities::int_to_string(timestep_number, 3) + ".vtk";
        std::ofstream output_fluid (filename_fluid.c_str());
        data_out_fluid.write_vtk (output_fluid);

    }
  
    template <int dim>
    void FullMovingMesh<dim>::compute_errors ()
    {
        const ComponentSelectFunction<dim>
        pressure_mask (dim, dim+1);
        const ComponentSelectFunction<dim>
        velocity_mask(std::make_pair(0, dim), dim+1);

        ExactFunction_pf<dim> exact_solution_pf;
        Vector<double> cellwise_errors (triangulation.n_active_cells());

        QTrapez<1>     q_trapez;
        QIterated<dim> quadrature (q_trapez, base_degree+2);

        VectorTools::integrate_difference (dof_handler_pf, solution_pf, exact_solution_pf,
                                           cellwise_errors, quadrature,
                                           VectorTools::L2_norm,
                                           &pressure_mask);
        const double p_l2_error = cellwise_errors.l2_norm();

        VectorTools::integrate_difference (dof_handler_pf, solution_pf, exact_solution_pf,
                                           cellwise_errors, quadrature,
                                           VectorTools::L2_norm,
                                           &velocity_mask);
        const double u_l2_error = cellwise_errors.l2_norm();

        std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
        << ",   ||e_u||_L2 = " << u_l2_error
        << std::endl;

        {
        ExactSolution_vf<dim> exact_solution_vf;
        Vector<double> cellwise_errors_vf (triangulation.n_active_cells());

        VectorTools::integrate_difference (dof_handler_vf, solution_vf, exact_solution_vf,
                                           cellwise_errors_vf, quadrature,
                                           VectorTools::L2_norm);
        const double l2_error = cellwise_errors_vf.l2_norm();


        std::cout << "Errors: ||e_vf||_L2 = " << l2_error
        << std::endl;
        }
        
    }
    
    template <int dim>
    void FullMovingMesh<dim>::print_mesh ()
    {
        std::ofstream out ("grid-" + Utilities::int_to_string(timestep_number, 3) + ".eps");
        GridOut grid_out;
        grid_out.write_eps (triangulation, out);
    }
    
    
    template <int dim>
    void FullMovingMesh<dim>::run ()
    {
        timestep_number = 0;
        time = 1.0;
        
        std::cout << "   Problem degrees: " << degree_vr << ", " << degree_pr << ", " << degree_pf << ", " << degree_vf << ", " << degree_phi << ", " << degree_T << "." << std::endl;
        
        make_initial_grid ();
        print_mesh ();
        setup_dofs_rock ();
        setup_dofs_fluid ();
        setup_dofs_phi ();
        setup_dofs_T ();
        
        InitialFunction_rock<dim> rock_function;
        rock_function.set_time (time);
        
        VectorTools::interpolate(dof_handler_rock, InitialFunction_rock<dim>(), solution_rock);
        VectorTools::interpolate(dof_handler_phi, InitialFunction_phi<dim>(), old_solution_phi);
        solution_phi = old_solution_phi;
    
        
        assemble_system_pf ();
        solve_fluid ();
        
        compute_errors ();
        output_results ();
    }
}

int main ()
{
    try
    {
        using namespace dealii;
        using namespace FullSolver;
        FullMovingMesh<2> flow_problem(degree_vr, degree_pr, degree_pf, degree_vf, degree_phi, degree_T);
        flow_problem.run ();
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

