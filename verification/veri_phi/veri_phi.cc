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
        const unsigned int base_degree = 1;
        const unsigned int degree_vr   = base_degree +1;
        const unsigned int degree_pr   = base_degree;
        const unsigned int degree_pf   = base_degree;
        const unsigned int degree_vf   = base_degree +1;
        const unsigned int degree_phi  = base_degree;
        const unsigned int degree_T    = base_degree;
        
        const int refinement_level = 5;
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        const double patm = 1.0;
        const double rho_f = 1.0;
        const double rho_r = 2.71;
        const double phi0 = 0.7;
        const double Re = 1e-24;
        const double Ri = 1e26;
        const double mu_r = 1.0;
        
        const double timestep_size = 0.01;
        const double A = 0.1/timestep_size;
        const double total_timesteps = 10;
    }
    
    using namespace data;
    template <int dim>
    class FullMovingMesh
    {
    public:
        FullMovingMesh (const unsigned int degree_vr, const unsigned int degree_pr,
                        const unsigned int degree_pf, const unsigned int degree_vf,
                        const unsigned int degree_phi, const unsigned int degree_T);
        
        ~FullMovingMesh ();

        void run ();
    private:
        void make_initial_grid ();
        void setup_dofs_rock ();
        void setup_dofs_fluid ();
        void setup_dofs_phi ();
        void setup_dofs_T ();
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
        
        const unsigned int        degree_vr;
        const unsigned int        degree_pr;
        Triangulation<dim>        triangulation;
        FESystem<dim>             fe_rock;
        DoFHandler<dim>           dof_handler_rock;
        ConstraintMatrix          constraints_rock;
        BlockSparsityPattern      sparsity_pattern_rock;
        BlockSparseMatrix<double> system_matrix_rock;
        BlockVector<double>       solution_rock;
        BlockVector<double>       old_solution_rock;
        BlockVector<double>       system_rhs_rock;
        
        const unsigned int        degree_pf;
        FESystem<dim>             fe_pf;
        DoFHandler<dim>           dof_handler_pf;
        ConstraintMatrix          constraints_pf;
        BlockSparsityPattern      sparsity_pattern_pf;
        BlockSparseMatrix<double> system_matrix_pf;
        BlockVector<double>       solution_pf;
        BlockVector<double>       old_solution_pf;
        BlockVector<double>       system_rhs_pf;
        
        const unsigned int        degree_vf;
        FESystem<dim>             fe_vf;
        DoFHandler<dim>           dof_handler_vf;
        ConstraintMatrix          constraints_vf;
        SparsityPattern           sparsity_pattern_vf;
        SparseMatrix<double>      system_matrix_vf;
        SparseMatrix<double>      mass_matrix_vf;
        Vector<double>            solution_vf;
        Vector<double>            system_rhs_vf;
        
        Vector<double>            solution_pdiff;
        
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
    FullMovingMesh<dim>::~FullMovingMesh()
    {
        dof_handler_phi.clear ();
    }

    
    template <int dim>
    class ExactFunction_pf : public Function<dim>
    {
    public: ExactFunction_pf () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double> &value) const;
    };

    template <int dim>
    class InitialFunction_rock : public Function<dim>
    {
    public: InitialFunction_rock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double> &value) const;
    };
    
    template <int dim>
    void InitialFunction_rock<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
        values(0) = p[0]+p[1];
        values(1) = p[0]*p[1];
        values(2) = 0;
    }
    
    template <int dim>
    void ExactFunction_pf<dim>::vector_value (const Point<dim> &p, Vector<double> &values) const
    {
        values(0) = (p[1]-1./3*p[1]*p[1]*p[1]); //ux = -grad pf x
        values(1) = p[0]*(1-p[1]*p[1]);  // uy = -grad pf y
        values(2) = -p[0]*(p[1]-1.0/3.0*p[1]*p[1]*p[1]); //p
    }

    template <int dim>
    class InitialFunction_phi : public Function<dim>
    {
    public: InitialFunction_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    };
    
    template <int dim>
    class ExtraRHS : public Function<dim>
    {
    public: ExtraRHS () : Function<dim>() {}
        virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    };


    template <int dim>
    double InitialFunction_phi<dim>::value (const Point<dim>  &p, const unsigned int /*component*/) const
    {
        
        const double time = this->get_time();
        const double phi = A*p[0]*exp(-p[1])*time;
        return phi;
    }
    
    template <int dim>
    double ExtraRHS<dim>::value (const Point<dim>  &p, const unsigned int /*component*/) const
    {
        const double time = this->get_time();
//        const double old_time = time - timestep_size;
//        const double pr = p[0]*p[0]*(1-p[1]);
//        const double pf = -p[0]*(p[1]-1.0/3.0*p[1]*p[1]*p[1]);
//        const double old_phi = phi0 + A*old_time*p[1]*p[0];
        const double dphidt = A*p[0]*exp(-p[1]);
        const double vrgrad = A*(p[1]+p[0])*exp(-p[1])*time - A*p[0]*p[0]*p[1]*exp(-p[1])*time;
        
        return dphidt + vrgrad;
    }
    
    template <int dim>
    class ExactSolution_phi : public Function<dim>
    {
    public: ExactSolution_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p, const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExactSolution_phi<dim>::value (const Point<dim>  &p, const unsigned int /*component*/) const
    {
        
        const double time = this->get_time();
        const double phi = A*p[0]*exp(-p[1])*time;
        return phi;
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
        
        GridGenerator::subdivided_hyper_rectangle (triangulation,subdivisions, bottom_left, top_right);
        
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
        constraints_rock.clear ();
        
        FEValuesExtractors::Vector velocities(0);
        FEValuesExtractors::Scalar pressure (dim);
        
        DoFTools::make_hanging_node_constraints (dof_handler_rock,
                                                 constraints_rock);
        
        constraints_rock.close ();
        
        std::vector<types::global_dof_index> dofs_per_block (2);
        DoFTools::count_dofs_per_block (dof_handler_rock, dofs_per_block, block_component);
        const unsigned int n_u = dofs_per_block[0],
        n_p = dofs_per_block[1];
        std::cout << "   Number of active cells: "
        << triangulation.n_active_cells()
        << std::endl
        << "   Number of degrees of freedom: "
        << dof_handler_rock.n_dofs()
        << " (" << n_u << '+' << n_p << ')'
        << std::endl;
        
        BlockDynamicSparsityPattern dsp (2,2);
        dsp.block(0,0).reinit (n_u, n_u);
        dsp.block(1,0).reinit (n_p, n_u);
        dsp.block(0,1).reinit (n_u, n_p);
        dsp.block(1,1).reinit (n_p, n_p);
        dsp.collect_sizes();
        DoFTools::make_sparsity_pattern (dof_handler_rock, dsp, constraints_rock, false);
        sparsity_pattern_rock.copy_from (dsp);
        
        system_matrix_rock.reinit (sparsity_pattern_rock);
        solution_rock.reinit (2);
        solution_rock.block(0).reinit (n_u);
        solution_rock.block(1).reinit (n_p);
        old_solution_rock.reinit (2);
        old_solution_rock.block(0).reinit (n_u);
        old_solution_rock.block(1).reinit (n_p);
        solution_rock.collect_sizes ();
        system_rhs_rock.reinit (2);
        system_rhs_rock.block(0).reinit (n_u);
        system_rhs_rock.block(1).reinit (n_p);
        system_rhs_rock.collect_sizes ();
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
        
        BlockDynamicSparsityPattern dsp_pf(2, 2);
        dsp_pf.block(0, 0).reinit (n_u, n_u);
        dsp_pf.block(1, 0).reinit (n_pf, n_u);
        dsp_pf.block(0, 1).reinit (n_u, n_pf);
        dsp_pf.block(1, 1).reinit (n_pf, n_pf);
        
        dsp_pf.collect_sizes ();
        DoFTools::make_sparsity_pattern (dof_handler_pf, dsp_pf, constraints_pf, false);
        
        sparsity_pattern_pf.copy_from(dsp_pf);
        system_matrix_pf.reinit (sparsity_pattern_pf);
        solution_pf.reinit (2);
        solution_pf.block(0).reinit (n_u);
        solution_pf.block(1).reinit (n_pf);
        old_solution_pf.reinit (2);
        old_solution_pf.block(0).reinit (n_u);
        old_solution_pf.block(1).reinit (n_pf);
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
    void FullMovingMesh<dim>::assemble_system_phi ()
    {
        mass_matrix_phi = 0;
        nontime_matrix_phi = 0;
        
        FullMatrix<double>                   cell_rhs_matrix;
        FullMatrix<double>                   cell_matrix;

        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>     quadrature_formula(degree_phi+3);
        QGauss<dim-1>   face_quadrature_formula(degree_phi+3);
        
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    | update_gradients |
                                     update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_phi (fe_phi, face_quadrature_formula,
                                              update_values | update_quadrature_points |
                                              update_JxW_values | update_normal_vectors);
        FEValues<dim> fe_values_vr (fe_rock, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_rock (fe_rock, face_quadrature_formula,
                                               update_values | update_quadrature_points |
                                               update_JxW_values | update_normal_vectors);
        
        const unsigned int dofs_per_cell   = fe_phi.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_phi.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values_phi.get_quadrature().size();
        
        cell_rhs_matrix.reinit (dofs_per_cell, dofs_per_cell);
        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<Tensor<1,dim>>       vr_values (n_q_points);
        std::vector<Tensor<1,dim>>       face_vr_values (n_face_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_phi.begin_active(),
        endc = dof_handler_phi.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            cell_rhs_matrix = 0;
            cell_matrix = 0;
            
            fe_values_phi.reinit(cell);
            fe_values_vr.reinit(vr_cell);
            
            fe_values_vr[velocities].get_function_values (solution_rock, vr_values);
            
            const double delta = 0.1 * cell->diameter ();
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    // (v_r.grad phi, psi+d*v_r.grad_psi)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                            cell_matrix(i,j) += fe_values_phi.shape_value(j,q_point)   *
                                             (fe_values_phi.shape_value(i,q_point)
                                              + delta * (vr_values[q_point] * //SDFEM
                                                         fe_values_phi.shape_grad(i,q_point))
                                              ) *
                                            fe_values_phi.JxW(q_point);
                        
                        
                            cell_rhs_matrix(i,j) += ((vr_values[q_point] * //Advection term DDt
                                                  fe_values_phi.shape_grad(j,q_point)   *
                                                  (fe_values_phi.shape_value(i,q_point)
                                                   + delta * (vr_values[q_point] * //SDFEM
                                                              fe_values_phi.shape_grad(i,q_point))
                                                   )) *
                                                 fe_values_phi.JxW(q_point));
                        
                    }
                }
            
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary()) // BC weak so need on LHS too
                {
                    fe_face_values_phi.reinit (cell, face);
                    fe_face_values_rock.reinit (vr_cell, face);
                    
                    fe_face_values_rock[velocities].get_function_values (solution_rock, face_vr_values);

                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        if (fe_face_values_phi.normal_vector(q_point) * face_vr_values[q_point] < 0)
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                            {
                                for (unsigned int j=0; j<dofs_per_cell; ++j)
                                    cell_rhs_matrix(i,j) -= (face_vr_values[q_point] *
                                                         fe_face_values_phi.normal_vector(q_point) *
                                                         fe_face_values_phi.shape_value(i,q_point) *
                                                         fe_face_values_phi.shape_value(j,q_point) *
                                                         fe_face_values_phi.JxW(q_point));
                            }
                }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                    nontime_matrix_phi.add (local_dof_indices[i],
                                            local_dof_indices[j],
                                            cell_rhs_matrix(i,j));
                    mass_matrix_phi.add (local_dof_indices[i],
                                            local_dof_indices[j],
                                            cell_matrix(i,j));
                }
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_rhs_phi ()
    {
        system_rhs_phi=0;
        nontime_rhs_phi=0;
        
        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(degree_phi+4);
        QGauss<dim-1> face_quadrature_formula(degree_phi+4);
        
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    |  update_gradients |
                                     update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values_phi (fe_phi, face_quadrature_formula,
                                              update_values | update_quadrature_points |
                                              update_JxW_values | update_normal_vectors);
        FEValues<dim> fe_values_pf (fe_pf, quadrature_formula,
                                    update_values    |  update_gradients |
                                    update_quadrature_points  |  update_JxW_values);
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                      update_values    |  update_gradients |
                                      update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values_rock (fe_rock, face_quadrature_formula,
                                               update_values | update_quadrature_points |
                                               update_JxW_values | update_normal_vectors);
        FEValues<dim> fe_values_T (fe_T, quadrature_formula,
                                   update_values    |  update_gradients |
                                   update_quadrature_points  |  update_JxW_values);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        const unsigned int dofs_per_cell   = fe_phi.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_phi.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values_phi.get_quadrature().size();
        
        cell_rhs.reinit (dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<double>              face_boundary_values (n_face_q_points);
        std::vector<Tensor<1,dim> >      face_vr_values (n_face_q_points);
        
        std::vector<Tensor<1,dim>>       vr_values (n_q_points);
        std::vector<double>              pr_values (n_q_points);
        std::vector<double>              pf_values (n_q_points);
        std::vector<double>              old_T_values (n_q_points);
        std::vector<double>              old_phi_values (n_q_points);
        std::vector<double>              rhs_values (n_q_points);
        std::vector<Tensor<1,dim>>       grad_pr_values (n_q_points);
        std::vector<Tensor<1,dim>>       grad_pf_values (n_q_points);
        std::vector<Tensor<1,dim>>       u_values (n_q_points);
        
        InitialFunction_phi<dim>  boundary_function;
        boundary_function.set_time(time);
        
        ExtraRHS<dim>       rhs_function;
        rhs_function.set_time(time);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_phi.begin_active(),
        endc = dof_handler_phi.end();
        typename DoFHandler<dim>::active_cell_iterator
        rock_cell = dof_handler_rock.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = dof_handler_pf.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        T_cell = dof_handler_T.begin_active();
        for (; cell!=endc; ++cell, ++rock_cell, ++pf_cell, ++T_cell)
        {
            cell_rhs = 0;
            
            fe_values_phi.reinit (cell);
            fe_values_pf.reinit (pf_cell);
            fe_values_rock.reinit (rock_cell);
            fe_values_T.reinit (T_cell);
            
            fe_values_pf[pressure].get_function_values (solution_pf, pf_values);
            fe_values_pf[velocities].get_function_values (solution_pf, u_values);
            fe_values_phi.get_function_values (old_solution_phi, old_phi_values);
            fe_values_T.get_function_values (old_solution_T, old_T_values);
            fe_values_rock[pressure].get_function_values (solution_rock, pr_values);
            fe_values_rock[pressure].get_function_gradients (solution_rock, grad_pr_values);
            fe_values_rock[velocities].get_function_values (solution_rock, vr_values);
            
            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            {
                
                const double chem_c = old_phi_values[q_point]*
                (1.0-old_phi_values[q_point]);
                rhs_function.value_list (fe_values_phi.get_quadrature_points(), rhs_values);
                
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                { // actual RHS
                    cell_rhs(i) += ((fe_values_phi.shape_value(i,q_point)
                                     + delta *(vr_values[q_point] * fe_values_phi.shape_grad(i,q_point))
                                     ) //SDFEM
                                    *(
                                      - 0*Re*Ri*chem_c/mu_r * (pr_values[q_point] - pf_values[q_point])
                                            + rhs_values[q_point]
                                      ) *
                                    fe_values_phi.JxW (q_point));
                    
                }
            }
            
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary())
                {
                    fe_face_values_phi.reinit (cell, face);
                    fe_face_values_rock.reinit (rock_cell, face);
                    
                    fe_face_values_rock[velocities].get_function_values (solution_rock, face_vr_values);
                    boundary_function.value_list (fe_face_values_phi.get_quadrature_points(), face_boundary_values);

                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                        if (fe_face_values_phi.normal_vector(q_point) * face_vr_values[q_point] < 0)
                            for (unsigned int i=0; i<dofs_per_cell; ++i)
                            { // Boundary values on rhs
                                for (unsigned int j=0; j<dofs_per_cell; ++j)
                                    cell_rhs(i) -= (face_vr_values[q_point] *
                                                    fe_face_values_phi.normal_vector(q_point) *
                                                    face_boundary_values[q_point]         *
                                                    fe_face_values_phi.shape_value(i,q_point) *
                                                    fe_face_values_phi.JxW(q_point));
//
                            }
                }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                nontime_rhs_phi(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_phi ()
    {
        mass_matrix_phi.vmult(system_rhs_phi, old_solution_phi);
        system_rhs_phi.add(timestep,nontime_rhs_phi);
        
        system_matrix_phi.copy_from(mass_matrix_phi);
        system_matrix_phi.add(timestep,nontime_matrix_phi);
        
        hanging_node_constraints_phi.condense (system_rhs_phi);
        hanging_node_constraints_phi.condense (system_matrix_phi);
        
//                        std::map<types::global_dof_index,double> boundary_values;
//                        ExactSolution_phi<dim> phi_boundary_values;
//                        phi_boundary_values.set_time(time);
//
//                        VectorTools::interpolate_boundary_values (dof_handler_phi, 1,
//                                                                  phi_boundary_values, boundary_values);
//                        MatrixTools::apply_boundary_values (boundary_values, system_matrix_phi,
//                                                            solution_phi, system_rhs_phi);
        
        std::cout << "   Solving for phi..." << std::endl;
        SparseDirectUMFPACK  phi_direct;
        phi_direct.initialize(system_matrix_phi);
        phi_direct.vmult (solution_phi, system_rhs_phi);
        hanging_node_constraints_phi.distribute (solution_phi);
        
    }
    
    template <int dim>
    void FullMovingMesh<dim>::output_results ()  const
    {
        
        //PHI
        DataOut<dim> data_out_phi;
        data_out_phi.attach_dof_handler(dof_handler_phi);
        data_out_phi.add_data_vector(solution_phi, "phi");
        data_out_phi.build_patches();
        
        const std::string filename_phi = "solution_phi-"
        + Utilities::int_to_string(timestep_number, 3) + ".vtk";
        std::ofstream output_phi(filename_phi.c_str());
        data_out_phi.write_vtk(output_phi);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::compute_errors ()
    {
        
        ExactSolution_phi<dim> exact_solution_phi;
        exact_solution_phi.set_time(time);
        
        Vector<double> difference_per_cell (triangulation.n_active_cells());
        
        const QTrapez<1>     q_trapez;
        const QIterated<dim> quadrature (q_trapez, base_degree+2);
        
        VectorTools::integrate_difference (dof_handler_phi,
                                           solution_phi,
                                           exact_solution_phi,
                                           difference_per_cell,
                                           quadrature,
                                           VectorTools::L2_norm);
        
        const double L2_error = difference_per_cell.l2_norm();
        
        std::cout << "Errors: L2 = " << L2_error << std::endl;
    }

    template <int dim>
    void FullMovingMesh<dim>::run ()
    {
        timestep_number = 0;
        time = 0.0;
        
        std::cout << "   Problem degrees: " << degree_vr << "," <<  degree_pr << "," <<  degree_pf << "," <<  degree_vf << "," <<  degree_phi << "." << std::endl;
        std::cout << "   Refinement level: " << refinement_level << "." << std::endl;
        
        make_initial_grid ();
        setup_dofs_rock ();
        setup_dofs_fluid ();
        setup_dofs_phi ();
        setup_dofs_T ();
        
        ExactFunction_pf<dim>   pf_function;
        pf_function.set_time (time);
        InitialFunction_rock<dim>   rock_function;
        rock_function.set_time (time);
        InitialFunction_phi<dim>    phi_function;
        phi_function.set_time(time);
        VectorTools::interpolate(dof_handler_pf, pf_function, solution_pf);
        VectorTools::interpolate(dof_handler_rock, rock_function, solution_rock);
        VectorTools::interpolate(dof_handler_phi, phi_function, old_solution_phi);
        solution_phi = old_solution_phi;
        
        std::cout << "===========================================" << std::endl;
        
        output_results();
        compute_errors ();
        
        timestep = timestep_size;
        
        while (timestep_number < total_timesteps)
        {
            std::cout << "   Moving to next time step" << std::endl
            << "===========================================" << std::endl;
            
            time += timestep;
            ++timestep_number;
            
            ExactFunction_pf<dim>   pf_function;
            pf_function.set_time (time);
            InitialFunction_rock<dim>   rock_function;
            rock_function.set_time (time);
            VectorTools::interpolate(dof_handler_pf, pf_function, solution_pf);
            VectorTools::interpolate(dof_handler_rock, rock_function, solution_rock);

            
            std::cout << "   Assembling at timestep number " << timestep_number
            << " from phi..." <<  std::endl << std::flush;
            
            assemble_system_phi ();
            assemble_rhs_phi ();
            solve_phi ();
            
            output_results ();
            compute_errors ();
            
            old_solution_phi = solution_phi;
        }
        std::cout << "===========================================" << std::endl;
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
