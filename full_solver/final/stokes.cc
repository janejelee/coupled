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
        const unsigned int degree_rock = base_degree;
        const unsigned int degree_pf = base_degree ;
        const unsigned int degree_vf = base_degree +1;
        const unsigned int degree_phi = base_degree;
        const unsigned int degree_T = base_degree;
        
        const int refinement_level = 3;
        const double top = 1.0;
        const double bottom = 0.0;
        const double left = 0.0;
        const double right = PI;
        const double lambda = 1.0;
        const double k = 1.0;
        const double rho_f = 1.0;
        const double c_f = 1.0;
        const double rho_r = 1.0;
        const double c_r = 1.0;
        const double phi0 = 0.7;
        const double vr2_constant = -bottom*bottom;
        const double C = 1;
        const double kappa = 1.0;
        const double Nu = 1.0;
        const double Re = 1.0;
        const double Ri = 1.0;
        const double chem_c = 1.0;
        const double mu_r = 1.0;
        
        const double timestep_size = 0.00001;
        const double total_timesteps = 5;
//        const int error_timestep = 1;
        
        ConvergenceTable                        convergence_table;
        ConvergenceTable                        convergence_table_rate;
    }
    
    using namespace data;
    
    template <int dim>
    class FullMovingMesh
    {
    public:
        FullMovingMesh (const unsigned int degree_rock, const unsigned int degree_pf, const unsigned int degree_vf, const unsigned int degree_phi, const unsigned int degree_T);
        ~FullMovingMesh ();
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
        void output_results () const;
        void move_mesh ();
        void print_mesh ();
        void compute_errors ();
        
        const unsigned int        degree_rock;
        Triangulation<dim>        triangulation;
        FESystem<dim>             fe_rock;
        DoFHandler<dim>           dof_handler_rock;
        ConstraintMatrix          constraints_rock;
        BlockSparsityPattern      sparsity_pattern_rock;
        BlockSparseMatrix<double> system_matrix_rock;
        BlockVector<double>       solution_rock;
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
        
        double time;
        double timestep;
        unsigned int timestep_number;
        
    };
    
    template <int dim>
    class ExactSolution_rock : public Function<dim>
    {
    public:
        ExactSolution_rock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactSolution_pf : public Function<dim>
    {
    public:
        ExactSolution_pf () : Function<dim>(dim) {}
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactSolution_vf : public Function<dim>
    {
    public:
        ExactSolution_vf () : Function<dim>(dim) {}
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactSolution_u : public Function<dim>
    {
    public:
        ExactSolution_u () : Function<dim>(dim) {}
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class ExactSolution_phi : public Function<dim>
    {
    public:
        ExactSolution_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    class ExactSolution_T : public Function<dim>
    {
    public:
        ExactSolution_T () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };

    template <int dim>
    void ExactSolution_rock<dim>::vector_value (const Point<dim> &p,
                                                Vector<double>   &values) const
    {
        values(0) = 0;
        values(1) = -p[1]*p[1];
        values(2) = p[1]*p[1]*p[1];
    }
    
    template <int dim>
    void ExactSolution_pf<dim>::vector_value (const Point<dim> &p,
                                              Vector<double>   &values) const
    {
        Assert (values.size() == dim+1,
                ExcDimensionMismatch (values.size(), dim+1));
        
        const double permeability = k;
        
        values(0) = 0.0;
        values(1) = lambda*rho_f*(1-p[1]*p[1])*permeability;
        values(2) = -rho_f*(p[1] - (1.0/3.0)*p[1]*p[1]*p[1]);
    }
    
    template <int dim>
    void ExactSolution_vf<dim>::vector_value (const Point<dim> &p,
                                              Vector<double>   &values) const
    {
        const double time = this->get_time();
        double phi = phi0 + C*time*exp(-p[1]*p[1]*p[1]);
        
        values(0) = 0.0;
        values(1) = -p[1]*p[1] - lambda*k*(rho_f - rho_f*(1-p[1]*p[1]))/phi;
    }

    template <int dim>
    void ExactSolution_u<dim>::vector_value (const Point<dim> &p,
                                              Vector<double>   &values) const
    {
        values(0) = 0.0;
        values(1) = lambda*k*rho_f*(1-p[1]*p[1]);
    }
    
    template <int dim>
    double ExactSolution_phi<dim>::value (const Point<dim>  &p,
                                          const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        return  phi0 + C*time*exp(-p[1]*p[1]*p[1]);
    }

    template <int dim>
    double ExactSolution_T<dim>::value (const Point<dim>  &p,
                                        const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        return  p[1]*p[1]*exp(-time);
    }

    
    // Extra terms you get on the right hand side from manufactured solutions
    template <int dim>
    class ExtraRHSRock : public Function<dim>
    {
    public:
        ExtraRHSRock () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    ExtraRHSRock<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        const double time = this->get_time();
        double phi = phi0 + C*time*exp(-p[1]*p[1]*p[1]);
        double dphidz = -3*p[1]*p[1]*C* time * exp(-p[1]*p[1]*p[1]);
        double vfz = -p[1]*p[1]*(1.0 + lambda*k*rho_f/phi);
        double dvf = -2*p[1]*(1.0 + lambda*k*rho_f/phi)
                            +3*pow(p[1],4.0)*lambda*k*rho_f*C*time*exp(-p[1]*p[1]*p[1])*(1.0/phi)*(1.0/phi);
        values(0) = 0;
        values(1) = -(4+3*p[1]*p[1])*(1-phi)
                        - C*time*exp(-p[1]*p[1]*p[1])*(12*pow(p[1],3.0)+ 3*pow(p[1],5.0));
        values(2) = -2*p[1]*(1-phi) - 3*C*time*pow(p[1],4.0)*exp(-p[1]*p[1]*p[1])
                        + (dphidz*vfz + phi*dvf);
    }
    
    template <int dim>
    class ExtraRHSpf : public Function<dim>
    {
    public:
        ExtraRHSpf () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExtraRHSpf<dim>::value (const Point<dim>   &p,
                                   const unsigned int) const
    {
        return 2*p[1]*(1 + lambda*k*rho_f);
    }
    
    template <int dim>
    class ExtraRHSphi : public Function<dim>
    {
    public:
        ExtraRHSphi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double
    ExtraRHSphi<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
    {
        const double time = this->get_time();
        return C*exp(-p[1]*p[1]*p[1]) + 3*C*time*pow(p[1],4.0)*exp(-p[1]*p[1]*p[1]) + pow(p[1],3.0)
                    + Re*Ri*chem_c/mu_r*( pow(p[1],3.0) - (-rho_f*(p[1]-1.0/3.0*pow(p[1],3.0)) ) );
    }
    
    template <int dim>
    class ExtraRHST : public Function<dim>
    {
    public:
        ExtraRHST () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double ExtraRHST<dim>::value (const Point<dim>   &p,
                                  const unsigned int) const
    {
        const double time = this->get_time();
        double T = p[1]*p[1]*exp(-time);
        double phi = phi0 + C*time*exp(-p[1]*p[1]*p[1]);
        double vfz = -p[1]*p[1]*(1 + lambda*k*rho_f/phi);
        double dvf = -2*p[1]*(1.0 + lambda*k*rho_f/phi) - 3*pow(p[1],4.0)*lambda*k*rho_f*C*time*
                            exp(-pow(p[1],3.0))*(1.0/phi)*(1.0/phi);
        
        return rho_f*c_f* (-p[1]*p[1]*exp(-time)*phi + C*exp(-p[1]*p[1]*p[1])*T ) +
            rho_r*c_r*(-p[1]*p[1]*exp(-time)*(1-phi) - T*C*exp(-p[1]*p[1]*p[1])) +
                rho_f*c_f* (-3*p[1]*p[1]*C*time*exp(-p[1]*p[1]*p[1])*vfz*T
                            + phi*dvf*T + phi*vfz*2*p[1]*exp(-time) ) +
                      rho_r*c_r* (-3*pow(p[1],4.0)*C*time*exp(-p[1]*p[1]*p[1])*T
                        -2*p[1]*(1.0-phi)*T -2*(1.0-phi)*p[1]*p[1]*p[1]*exp(-time))
                            - kappa/(phi0*Nu)*2*exp(-time);
    }
    
    // n.(pI-2e(u)) for the top
    template <int dim>
    class RockTopStress : public Function<dim>
    {
    public:
        RockTopStress () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    RockTopStress<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        const double time = this->get_time();
        double phi = phi0 + C*time*exp(-p[1]*p[1]*p[1]);
        values(0) = 0;
        values(1) = -(1-phi)*(p[1]*p[1]*p[1] + 4*p[1]);
        values(2) = 0;
    }
    
    // n.(pI-2e(u)) for the top
    template <int dim>
    class RockBottomStress : public Function<dim>
    {
    public:
        RockBottomStress () : Function<dim>(dim+1) {}
        virtual void vector_value (const Point<dim> &p, Vector<double>   &value) const;
    };
    
    template <int dim>
    void
    RockBottomStress<dim>::vector_value (const Point<dim> &p, Vector<double>   &values) const
    {
        const double time = this->get_time();
        double phi = phi0 + C*time*exp(-p[1]*p[1]*p[1]);
        values(0) = 0;
        values(1) = (1-phi)*(p[1]*p[1]*p[1] + 4*p[1]);
        values(2) = 0;
    }
    
    template <int dim>
    class HeatFluxFunction : public Function<dim>
    {
    public: HeatFluxFunction () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double HeatFluxFunction<dim>::value (const Point<dim>  &p,
                                         const unsigned int /*component*/) const
    {
        const double time = this->get_time();
        return -2*kappa*p[1]*exp(-time)/(phi0*Nu);
    }
    
    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
        PressureBoundaryValues () : Function<dim>(1) {}
        
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    double PressureBoundaryValues<dim>::value (const Point<dim>  &p,
                                               const unsigned int /*component*/) const
    {
        // This is the dirichlet condition for the top
        return -rho_f*(p[1]-1./3*p[1]*p[1]*p[1]); // last term added to avoid warnings on work comp
    }
    
    template <int dim>
    class InitialFunction_phi : public Function<dim>
    {
    public:
        InitialFunction_phi () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    class InitialFunction_vf : public Function<dim>
    {
    public:
        InitialFunction_vf () : Function<dim>(dim) {}
        
        virtual void vector_value (const Point<dim> &p,
                                   Vector<double>   &value) const;
    };
    
    template <int dim>
    class InitialFunction_T : public Function<dim>
    {
    public:
        InitialFunction_T () : Function<dim>() {}
        virtual double value (const Point<dim>   &p,
                              const unsigned int  component = 0) const;
    };
    
    template <int dim>
    void InitialFunction_vf<dim>::vector_value (const Point<dim> &p,
                                              Vector<double>   &values) const
    {
        const double time = 0.0;
        double phi = phi0 + C*time*exp(-p[1]*p[1]*p[1]);
        
        values(0) = 0.0;
        values(1) = -p[1]*p[1] - lambda*k*(rho_f - rho_f*(1-p[1]*p[1]))/phi;
    }

    template <int dim>
    double InitialFunction_phi<dim>::value (const Point<dim>  &p,
                                        const unsigned int /*component*/) const
    {
        return phi0;
    }

    template <int dim>
    double InitialFunction_T<dim>::value (const Point<dim>  &p,
                                            const unsigned int /*component*/) const
    {
        return p[1]*p[1];
    }
    
    template <int dim>
    class KInverse : public TensorFunction<2,dim>
    {
    public:
        KInverse () : TensorFunction<2,dim>() {}
        
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<2,dim> >    &values) const;
    };
    
    template <int dim>
    void
    KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                               std::vector<Tensor<2,dim> >    &values) const
    {
        
        Assert (points.size() == values.size(),
                ExcDimensionMismatch (points.size(), values.size()));
        
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();
            const double permeability = k;
            
            for (unsigned int d=0; d<dim; ++d)
                values[p][d][d] = 1./permeability;
        }
    }
    
    template <int dim>
    class K : public TensorFunction<2,dim>
    {
    public:
        K () : TensorFunction<2,dim>() {}
        
        virtual void value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<2,dim> >    &values) const;
    };
    
    template <int dim>
    void
    K<dim>::value_list (const std::vector<Point<dim> > &points,
                        std::vector<Tensor<2,dim> >    &values) const
    {
        Assert (points.size() == values.size(),
                ExcDimensionMismatch (points.size(), values.size()));
        
        for (unsigned int p=0; p<points.size(); ++p)
        {
            values[p].clear ();
            Assert (points.size() == values.size(),
                    ExcDimensionMismatch (points.size(), values.size()));
            
            for (unsigned int p=0; p<points.size(); ++p)
            {
                values[p].clear ();/*
                                    const double distance_to_flowline
                                    = std::fabs(points[p][1]-0.2*std::sin(10*points[p][0]));*/
                const double permeability = k;
                
                for (unsigned int d=0; d<dim; ++d)
                    values[p][d][d] = permeability;
            }
        }
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
    FullMovingMesh<dim>::FullMovingMesh (const unsigned int degree_rock, const unsigned int degree_pf, const unsigned int degree_vf, const unsigned int degree_phi, const unsigned int degree_T)
    :
    degree_rock (degree_rock),
    triangulation (Triangulation<dim>::maximum_smoothing),
    fe_rock (FE_Q<dim>(degree_rock+1), dim,
             FE_Q<dim>(degree_rock), 1),
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
    FullMovingMesh<dim>::~FullMovingMesh ()
    {
        dof_handler_phi.clear ();
        dof_handler_T.clear ();
    }
    
    template <int dim>
    void FullMovingMesh<dim>::make_initial_grid ()
    {
        std::vector<unsigned int> subdivisions (dim, 1);
        subdivisions[0] = 4;
        
        const Point<dim> bottom_left = (dim == 2 ?
                                        Point<dim>( left, bottom) :
                                        Point<dim>(-2,0,-1));
        const Point<dim> top_right   = (dim == 2 ?
                                        Point<dim>( right, top) :
                                        Point<dim>(0,1,0));
        
        GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                   subdivisions, bottom_left, top_right);
        
        for (typename Triangulation<dim>::active_cell_iterator
             cell = triangulation.begin_active();
             cell != triangulation.end(); ++cell)
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->center()[dim-1] ==  top)
                    cell->face(f)->set_all_boundary_ids(1);
                else if (cell->face(f)->center()[dim-1] ==  bottom)
                    cell->face(f)->set_all_boundary_ids(2);
        
        triangulation.refine_global (refinement_level);
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
        
        // fluid motion numbering
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
        solution_pf.collect_sizes ();
        
        system_rhs_pf.reinit (2);
        system_rhs_pf.block(0).reinit (n_u);
        system_rhs_pf.block(1).reinit (n_pf);
        system_rhs_pf.collect_sizes ();
        
        system_matrix_vf.clear ();
        constraints_vf.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_vf,
                                                 constraints_vf);
        
        constraints_vf.close();
        DynamicSparsityPattern dsp_vf(dof_handler_vf.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_vf,
                                        dsp_vf,
                                        constraints_vf,
                                        /*keep_constrained_dofs = */ true);
        sparsity_pattern_vf.copy_from(dsp_vf);
        mass_matrix_vf.reinit (sparsity_pattern_vf);
        system_matrix_vf.reinit (sparsity_pattern_vf);
        solution_vf.reinit (dof_handler_vf.n_dofs());
        system_rhs_vf.reinit (dof_handler_vf.n_dofs());
        
        MatrixCreator::create_mass_matrix(dof_handler_vf,
                                          QGauss<dim>(degree_vf+2),
                                          system_matrix_vf);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_phi ()
    {
        dof_handler_phi.distribute_dofs (fe_phi);
        hanging_node_constraints_phi.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_phi,
                                                 hanging_node_constraints_phi);
        hanging_node_constraints_phi.close ();
        std::cout << "   Number of degrees of freedom in porosity problem: "
        << dof_handler_phi.n_dofs()
        << std::endl;
        DynamicSparsityPattern dsp_phi(dof_handler_phi.n_dofs(), dof_handler_phi.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler_phi,
                                        dsp_phi,
                                        hanging_node_constraints_phi,
                                        /*keep_constrained_dofs = */ true);
        sparsity_pattern_phi.copy_from (dsp_phi);
        system_matrix_phi.reinit (sparsity_pattern_phi);
        mass_matrix_phi.reinit (sparsity_pattern_phi);
        nontime_matrix_phi.reinit (sparsity_pattern_phi);
        solution_phi.reinit (dof_handler_phi.n_dofs());
        old_solution_phi.reinit(dof_handler_phi.n_dofs());
        system_rhs_phi.reinit (dof_handler_phi.n_dofs());
        nontime_rhs_phi.reinit (dof_handler_phi.n_dofs());
        
        MatrixCreator::create_mass_matrix(dof_handler_phi,
                                          QGauss<dim>(degree_phi+4),
                                          mass_matrix_phi);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::setup_dofs_T ()
    {
        dof_handler_T.distribute_dofs (fe_T);
        DoFRenumbering::Cuthill_McKee (dof_handler_T);
        
        hanging_node_constraints_T.clear ();
        DoFTools::make_hanging_node_constraints (dof_handler_T,
                                                 hanging_node_constraints_T);
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
    void FullMovingMesh<dim>::apply_BC_rock ()
    {
        constraints_rock.clear ();
        
        FEValuesExtractors::Vector velocities(0);
        FEValuesExtractors::Scalar pressure (dim);
        
        DoFTools::make_hanging_node_constraints (dof_handler_rock,constraints_rock);
        
        // These are the conditions for the side boundary ids 0 (no flux)
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert (0);
        VectorTools::compute_no_normal_flux_constraints (dof_handler_rock, 0,
                                                         no_normal_flux_boundaries, constraints_rock);
        
        constraints_rock.close ();
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_rock ()
    {
        system_matrix_rock=0;
        system_rhs_rock=0;
        QGauss<dim>   quadrature_formula(degree_rock+2);
        QGauss<dim-1> face_quadrature_formula(degree_rock+2);
        
        
        FEValues<dim> fe_values_rock (fe_rock, quadrature_formula,
                                      update_values    |
                                      update_quadrature_points  |
                                      update_JxW_values |
                                      update_gradients);
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    |
                                     update_quadrature_points  |
                                     update_JxW_values |
                                     update_gradients);
        FEValues<dim> fe_values_vf (fe_vf, quadrature_formula,
                                     update_values    |
                                     update_quadrature_points  |
                                     update_JxW_values |
                                     update_gradients);
        FEFaceValues<dim> fe_face_values_rock ( fe_rock, face_quadrature_formula,
                                               update_values | update_normal_vectors |
                                               update_quadrature_points |update_JxW_values   );
        
        const unsigned int   dofs_per_cell   = fe_rock.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        ExtraRHSRock<dim>          right_hand_side;
        right_hand_side.set_time(time);
        std::vector<Vector<double> >     rhs_values (n_q_points, Vector<double>(dim+1));
        
        std::vector<double>                 phi_values (n_q_points);
        std::vector<Tensor<1,dim>>          grad_phi_values (n_q_points);
        std::vector<Vector<double> >        vf_values(n_q_points, Vector<double>(dim));
        std::vector<double>                 div_vf_values (n_q_points);
        
        RockTopStress<dim>        topstress;
        topstress.set_time(time);
        RockBottomStress<dim>     bottomstress;
        bottomstress.set_time(time);
        std::vector<Vector<double> >      topstress_values (n_face_q_points, Vector<double>(dim+1));
        std::vector<Vector<double> >      bottomstress_values (n_face_q_points, Vector<double>(dim+1));
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        std::vector<SymmetricTensor<2,dim> > symgrad_phi_u (dofs_per_cell);
        std::vector<double>                  div_phi_u   (dofs_per_cell);
        std::vector<double>                  phi_p       (dofs_per_cell);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_rock.begin_active(),
        endc = dof_handler_rock.end();
        typename DoFHandler<dim>::active_cell_iterator
        phi_cell = dof_handler_phi.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        vf_cell = dof_handler_vf.begin_active();
        for (; cell!=endc; ++cell, ++phi_cell, ++vf_cell)
        {
            fe_values_rock.reinit (cell);
            fe_values_phi.reinit (phi_cell);
            fe_values_vf.reinit (vf_cell);
            
            local_matrix = 0;
            local_rhs = 0;
            right_hand_side.vector_value_list(fe_values_rock.get_quadrature_points(),
                                                        rhs_values);
            fe_values_phi.get_function_values (solution_phi, phi_values);
            fe_values_phi.get_function_gradients (solution_phi, grad_phi_values);
            fe_values_vf.get_function_values (solution_vf, vf_values);
            fe_values_vf[velocities].get_function_divergences (solution_vf, div_vf_values);

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                {
                    symgrad_phi_u[k] = fe_values_rock[velocities].symmetric_gradient (k, q);
                    div_phi_u[k]     = fe_values_rock[velocities].divergence (k, q);
                    phi_p[k]         = fe_values_rock[pressure].value (k, q);
                }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    Tensor<1,dim> vf;
                    for (unsigned int d=0; d<dim; ++d)
                        vf[d] = vf_values[q](d);
                    
                    for (unsigned int j=0; j<=i; ++j)
                    { // matrix assembly
                        local_matrix(i,j) += (- 2 * (1-phi_values[q])*
                                              (symgrad_phi_u[i] * symgrad_phi_u[j])
                                              + (1-phi_values[q])*div_phi_u[i] * phi_p[j]
                                              + (1-phi_values[q])*phi_p[i] * div_phi_u[j])
                                                * fe_values_rock.JxW(q);
                    }
                    
                    const unsigned int component_i =
                    fe_rock.system_to_component_index(i).first;
                    
                    // Add the extra terms on RHS
//                    if (timestep_number == 0)
//                        {
//                            local_rhs(i) +=  (fe_values_rock.shape_value(i,q) *
//                                              initial_rhs_values[q](component_i)
//                                              )*
//                                            fe_values_rock.JxW(q);
//                        }
//                    else
//                        {
                            local_rhs(i) +=  (fe_values_rock.shape_value(i,q) *
                                              rhs_values[q](component_i)
                                              - (grad_phi_values[q]*vf + phi_values[q]*div_vf_values[q])
//                                              - (grad_phi_values[q]* vf + phi_values[q]* div_vf_values[q])
                                              * phi_p[i]
                                              )*
                                                fe_values_rock.JxW(q);
//                        }
                    
                    

                }
            }
            
            //            //Neumann Stress conditions on top boundary
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary()
                    &&
                    (cell->face(face_number)->boundary_id() == 1))
                {
                    fe_face_values_rock.reinit (cell, face_number);
                    topstress.vector_value_list(fe_face_values_rock.get_quadrature_points(), topstress_values);
                    
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            const unsigned int component_i = fe_rock.system_to_component_index(i).first;
                            
                            local_rhs(i) += (-topstress_values[q_point](component_i)*
                                             fe_face_values_rock.
                                             shape_value(i,q_point) *
                                             fe_face_values_rock.JxW(q_point));
                        }
                    }
                }
            
            // Neumann Stress conditions on bottom boundary
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary()
                    &&
                    (cell->face(face_number)->boundary_id() == 2))
                {
                    
                    fe_face_values_rock.reinit (cell, face_number);
                    bottomstress.vector_value_list
                    (fe_face_values_rock.get_quadrature_points(),
                     bottomstress_values);
                    
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            const unsigned int component_i = fe_rock.system_to_component_index(i).first;
                            local_rhs(i) += (-bottomstress_values[q_point](component_i)*
                                             fe_face_values_rock.
                                             shape_value(i,q_point) *
                                             fe_face_values_rock.JxW(q_point));
                        }
                    }
                }
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=i+1; j<dofs_per_cell; ++j)
                    local_matrix(i,j) = local_matrix(j,i);
            
            cell->get_dof_indices (local_dof_indices);
            
            constraints_rock.distribute_local_to_global (local_matrix, local_rhs,
                                                         local_dof_indices,
                                                         system_matrix_rock, system_rhs_rock);
        }
        
        std::map<types::global_dof_index, double> vr_determination;
        {
            types::global_dof_index n_dofs = dof_handler_rock.n_dofs();
            std::vector<bool> componentVector(dim + 1, false);
            componentVector[dim-1] = true;
            
            std::vector<bool> selected_dofs(n_dofs);
            std::set< types::boundary_id > boundary_ids;
            boundary_ids.insert(2);
            
            DoFTools::extract_boundary_dofs(dof_handler_rock, ComponentMask(componentVector),
                                            selected_dofs, boundary_ids);
            
            for (types::global_dof_index i = 0; i < n_dofs; i++)
            {
                if (selected_dofs[i]) vr_determination[i] = vr2_constant;
            }
        }
        
        MatrixTools::apply_boundary_values(vr_determination,
                                           system_matrix_rock, solution_rock, system_rhs_rock);
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
        
        const unsigned int   dofs_per_cell   = fe_pf.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();
        
        FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
        Vector<double>       local_rhs (dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
        
        const ExtraRHSpf<dim>             pf_rhs_values;
        const PressureBoundaryValues<dim>       pressure_boundary_values;
        const KInverse<dim>               k_inverse;
        const K<dim>                         k;
        
        std::vector<double> div_vr_values (n_q_points);
        std::vector<double> extra_pf_rhs (n_q_points);
        std::vector<double> boundary_values (n_face_q_points);
        std::vector<Tensor<2,dim> > k_inverse_values (n_q_points);
        std::vector<Tensor<2,dim> > k_values (n_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_pf.begin_active(),
        endc = dof_handler_pf.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            fe_values_pf.reinit (cell);
            fe_values_rock.reinit(vr_cell);
            
            local_matrix = 0;
            local_rhs = 0;
            
            k_inverse.value_list (fe_values_pf.get_quadrature_points(),
                                  k_inverse_values);
            k.value_list (fe_values_pf.get_quadrature_points(),
                          k_values);
            fe_values_rock[velocities].get_function_divergences (solution_rock, div_vr_values);
            pf_rhs_values.value_list (fe_values_pf.get_quadrature_points(), extra_pf_rhs);
            
            for (unsigned int q=0; q<n_q_points; ++q)
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
                        
                        local_matrix(i,j) += ( 1./lambda * phi_i_u * k_inverse_values[q] * phi_j_u
                                              - div_phi_i_u * phi_j_p
                                              - phi_i_p * div_phi_j_u)
                        * fe_values_pf.JxW(q);
                    }
                    
                    local_rhs(i) += phi_i_p *
                    (div_vr_values[q] + extra_pf_rhs[q]) *
                    fe_values_pf.JxW(q);
                    // BE CAREFUL HERE ONCE K is not constant or 1 anymore. have taken div of the constants as zero
                }
            
            for (unsigned int face_no=0;
                 face_no<GeometryInfo<dim>::faces_per_cell;
                 ++face_no)
                if (cell->face(face_no)->at_boundary()
                    &&
                    (cell->face(face_no)->boundary_id() == 1)) // Basin top has boundary id 1
                {
                    fe_face_values_pf.reinit (cell, face_no);
                    // DIRICHLET CONDITION FOR TOP. pf = pr at top of basin
                    
                    pressure_boundary_values
                    .value_list (fe_face_values_pf.get_quadrature_points(),
                                 boundary_values);
                    
                    for (unsigned int q=0; q<n_face_q_points; ++q)
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            local_rhs(i) += -( fe_face_values_pf[velocities].value (i, q) *
                                              fe_face_values_pf.normal_vector(q) *
                                              boundary_values[q] *
                                              fe_face_values_pf.JxW(q));
                }
            
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    system_matrix_pf.add (local_dof_indices[i],
                                          local_dof_indices[j],
                                          local_matrix(i,j));
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                system_rhs_pf(local_dof_indices[i]) += local_rhs(i);
        }
        
        // NOW NEED TO STRONGLY IMPOSE THE FLUX CONDITIONS BOTH ON SIDES AND BOTTOM
        {std::map<types::global_dof_index, double> boundary_values_flux;
            {
                types::global_dof_index n_dofs = dof_handler_pf.n_dofs();
                std::vector<bool> componentVector(dim + 1, true); // condition is on pressue
                // setting flux value for the sides at 0 ON THE PRESSURE
                componentVector[dim] = false;
                std::vector<bool> selected_dofs(n_dofs);
                std::set< types::boundary_id > boundary_ids;
                boundary_ids.insert(0);
                
                DoFTools::extract_boundary_dofs(dof_handler_pf, ComponentMask(componentVector),
                                                selected_dofs, boundary_ids);
                
                for (types::global_dof_index i = 0; i < n_dofs; i++) {
                    if (selected_dofs[i]) boundary_values_flux[i] = 0.0; // Side boudaries have flux 0 on pressure
                }
            }
            // Apply the conditions
            MatrixTools::apply_boundary_values(boundary_values_flux,
                                               system_matrix_pf, solution_pf, system_rhs_pf);
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
                { local_rhs(i) +=    ( vr_values[q_point][component_i]
                                      + (1.0/phi_values[q_point])*
                                      (u_values[q_point][component_i]
                                       - lambda*k*rho_f*unitz_values[q_point][component_i])
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
    void FullMovingMesh<dim>::assemble_system_phi ()
    {
        system_matrix_phi = 0;
        nontime_matrix_phi = 0;
        
        FullMatrix<double>                   cell_matrix;
        
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(degree_phi+3);
        QGauss<dim-1> face_quadrature_formula(degree_phi+3);
        
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                     update_values    | update_gradients |
                                     update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_phi (fe_phi, face_quadrature_formula,
                                              update_values | update_quadrature_points |
                                              update_JxW_values | update_normal_vectors);
        FEValues<dim> fe_values_vr (fe_rock, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        
        const unsigned int dofs_per_cell   = fe_phi.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_phi.get_quadrature().size();
        
        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<Tensor<1,dim>>       vr_values (n_q_points);
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_phi.begin_active(),
        endc = dof_handler_phi.end();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        for (; cell!=endc; ++cell, ++vr_cell)
        {
            cell_matrix = 0;
            
            fe_values_phi.reinit(cell);
            fe_values_vr.reinit(vr_cell);
            
            fe_values_vr[velocities].get_function_values (solution_rock, vr_values);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    // (v_r.grad phi, psi+d*v_r.grad_psi)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        cell_matrix(i,j) += ((vr_values[q_point] *
                                              fe_values_phi.shape_grad(j,q_point)   *
                                              (fe_values_phi.shape_value(i,q_point)
//                                                       + delta * (advection_directions[q_point] *
//                                                                fe_values.shape_grad(i,q_point))
                                               )) *
                                             fe_values_phi.JxW(q_point));
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                    nontime_matrix_phi.add (local_dof_indices[i],
                                            local_dof_indices[j],
                                            cell_matrix(i,j));
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
        
        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        ExtraRHSphi<dim>  right_hand_side;
        right_hand_side.set_time(time);
        
        const unsigned int dofs_per_cell   = fe_phi.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_phi.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values_phi.get_quadrature().size();
        
        cell_rhs.reinit (dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<double>         extra_phi_values (n_q_points);
        std::vector<double>         face_boundary_values (n_face_q_points);
        std::vector<Tensor<1,dim> > face_advection_directions (n_face_q_points);
        
        std::vector<Tensor<1,dim>>       vr_values (n_q_points);
        std::vector<double>              pr_values (n_q_points);
        std::vector<double>              pf_values (n_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_phi.begin_active(),
        endc = dof_handler_phi.end();
        typename DoFHandler<dim>::active_cell_iterator
        rock_cell = dof_handler_rock.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        pf_cell = dof_handler_pf.begin_active();
        for (; cell!=endc; ++cell, ++rock_cell, ++pf_cell)
        {
            cell_rhs = 0;
            
            fe_values_phi.reinit (cell);
            fe_values_pf.reinit (pf_cell);
            fe_values_rock.reinit (rock_cell);
            
            fe_values_pf[pressure].get_function_values (solution_pf, pf_values);
            fe_values_rock[pressure].get_function_values (solution_rock, pr_values);
            fe_values_rock[velocities].get_function_values (solution_rock, vr_values);
            
            right_hand_side.value_list (fe_values_phi.get_quadrature_points(),
                                        extra_phi_values);
            
            //            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += ((fe_values_phi.shape_value(i,q_point)
                                     //     + delta *(advection_directions[q_point] *
//                                            fe_values.shape_grad(i,q_point))
                                     )
                                    *(-Re*Ri*chem_c/mu_r*(pr_values[q_point] - pf_values[q_point]) +
                                         extra_phi_values[q_point]) *
                                    fe_values_phi.JxW (q_point));
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                nontime_rhs_phi(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_system_T ()
    {
        system_matrix_T = 0;
        mass_matrix_T = 0;
        mass_matrix_T_old = 0;
        nontime_matrix_T = 0;
        
        FullMatrix<double>                   cell_matrix;
        FullMatrix<double>                   cell_matrix_old;
        FullMatrix<double>                   cell_matrix_nontime;
        
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(degree_T+3);
        QGauss<dim-1> face_quadrature_formula(degree_T+3);
        
        FEValues<dim> fe_values_T (fe_T, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEValues<dim> fe_values_phi (fe_phi, quadrature_formula,
                                   update_values    |  update_gradients |
                                   update_quadrature_points  |  update_JxW_values);
        FEValues<dim> fe_values_vr (fe_rock, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);
        FEValues<dim> fe_values_vf (fe_vf, quadrature_formula,
                                    update_values    | update_gradients |
                                    update_quadrature_points  | update_JxW_values);

        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);
        
        const unsigned int dofs_per_cell   = fe_T.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_T.get_quadrature().size();
    
        cell_matrix.reinit (dofs_per_cell, dofs_per_cell);
        cell_matrix_nontime.reinit (dofs_per_cell, dofs_per_cell);
        cell_matrix_old.reinit (dofs_per_cell, dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<double>            phi_values (n_q_points);
        std::vector<double>            old_phi_values (n_q_points);
        std::vector<Tensor<1,dim>>     vr_values (n_q_points);
        std::vector<Vector<double> >   vf_values(n_q_points, Vector<double>(dim));
        std::vector<double>            div_vr_values (n_q_points);
        std::vector<double>            div_vf_values (n_q_points);
        std::vector<Tensor<1,dim>>     grad_phi_values (n_q_points);

        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_T.begin_active(),
        endc = dof_handler_T.end();
        typename DoFHandler<dim>::active_cell_iterator
        phi_cell = dof_handler_phi.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        vr_cell = dof_handler_rock.begin_active();
        typename DoFHandler<dim>::active_cell_iterator
        vf_cell = dof_handler_vf.begin_active();
        for (; cell!=endc; ++cell, ++phi_cell, ++vr_cell, ++vf_cell)
        {
            cell_matrix = 0;
            cell_matrix_old = 0;
            cell_matrix_nontime = 0;
            
            fe_values_T.reinit (cell);
            fe_values_phi.reinit (phi_cell);
            fe_values_vr.reinit (vr_cell);
            fe_values_vf.reinit (vf_cell);
            fe_values_phi.get_function_values (old_solution_phi, old_phi_values);
            fe_values_phi.get_function_values (solution_phi, phi_values);
            fe_values_phi.get_function_gradients (solution_phi, grad_phi_values);
            fe_values_vr[velocities].get_function_values (solution_rock, vr_values);
            fe_values_vr[velocities].get_function_divergences (solution_rock, div_vr_values);
            fe_values_vf.get_function_values (solution_vf, vf_values);
            fe_values_vf[velocities].get_function_divergences (solution_vf, div_vf_values);
            
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    Tensor<1,dim> vf;
                    for (unsigned int d=0; d<dim; ++d)
                        vf[d] = vf_values[q_point](d);
                    
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        cell_matrix_nontime(i,j) += ( (
                                                       rho_f*c_f*
                                                       (grad_phi_values[q_point]* vf +
                                                        phi_values[q_point] * div_vf_values[q_point] )
                                                       +
                                                      rho_r*c_r*
                                                       ( -grad_phi_values[q_point]* vr_values[q_point] +
                                                      (1.0 - phi_values[q_point])* div_vr_values[q_point] )
                                                       )*
                                                      fe_values_T.shape_value(i,q_point) *
                                                      fe_values_T.shape_value(j,q_point)
                                                      +
                                                      ( rho_f * c_f * phi_values[q_point] * vf  +
                                                       rho_r*c_r*
                                                      (1.0 - phi_values[q_point])* vr_values[q_point] )*
                                                      fe_values_T.shape_grad(j,q_point)   *
                                                      fe_values_T.shape_value(i,q_point)
                                                      +
                                                      kappa/(phi0*Nu)
                                                      *fe_values_T.shape_grad(i,q_point) *
                                                      fe_values_T.shape_grad(j,q_point)
                                                      ) *
                                                     fe_values_T.JxW(q_point);
                        
                        cell_matrix_old(i,j) += ( rho_f*c_f*(old_phi_values[q_point]) +
                                                 rho_r*c_r*(1.0 - old_phi_values[q_point])
                                                 )*
                                                    fe_values_T.shape_value(i,q_point)*
                                                    fe_values_T.shape_value(j,q_point)*
                                                    fe_values_T.JxW(q_point);
                        
                        cell_matrix(i,j) += ( rho_f*c_f*(phi_values[q_point]) +
                                             rho_r*c_r*(1.0 - phi_values[q_point])
                                             )*
                                                fe_values_T.shape_value(i,q_point)*
                                                fe_values_T.shape_value(j,q_point)*
                                                fe_values_T.JxW(q_point);
                    }
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                    nontime_matrix_T.add (local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix_nontime(i,j));
                    mass_matrix_T_old.add (local_dof_indices[i],
                                         local_dof_indices[j],
                                         cell_matrix_old(i,j));
                    mass_matrix_T.add (local_dof_indices[i],
                                     local_dof_indices[j],
                                     cell_matrix(i,j));
                }
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::assemble_rhs_T ()
    {
        system_rhs_T=0;
        nontime_rhs_T=0;
        
        Vector<double>                       cell_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        
        QGauss<dim>  quadrature_formula(degree_T+4);
        QGauss<dim-1> face_quadrature_formula(degree_T+4);
        
        FEValues<dim> fe_values_T (fe_T, quadrature_formula,
                                 update_values    |  update_gradients |
                                 update_quadrature_points  |  update_JxW_values);
        FEFaceValues<dim> fe_face_values_T (fe_T, face_quadrature_formula,
                                          update_values | update_quadrature_points |
                                          update_JxW_values | update_normal_vectors);
        
        ExtraRHST<dim>  right_hand_side;
        right_hand_side.set_time(time);
        
        const unsigned int dofs_per_cell   = fe_T.dofs_per_cell;
        const unsigned int n_q_points      = fe_values_T.get_quadrature().size();
        const unsigned int n_face_q_points = fe_face_values_T.get_quadrature().size();
        
        cell_rhs.reinit (dofs_per_cell);
        local_dof_indices.resize(dofs_per_cell);
        
        std::vector<double>         rhs_values (n_q_points);
        std::vector<Tensor<1,dim> > advection_directions (n_q_points);
        std::vector<double>         face_boundary_values (n_face_q_points);
        std::vector<Tensor<1,dim> > face_advection_directions (n_face_q_points);
        
        HeatFluxFunction<dim>     heat_flux;
        heat_flux.set_time(time);
        std::vector<double> heatflux_values (n_q_points);
        
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_T.begin_active(),
        endc = dof_handler_T.end();
        for (; cell!=endc; ++cell)
        {
            cell_rhs = 0;
            
            fe_values_T.reinit (cell);
            right_hand_side.value_list (fe_values_T.get_quadrature_points(),
                                        rhs_values);
            heat_flux.value_list (fe_values_T.get_quadrature_points(),
                                  heatflux_values);
            
            //            const double delta = 0.1 * cell->diameter ();
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                {
                    cell_rhs(i) += ((fe_values_T.shape_value(i,q_point)
                                     //                                     +
                                     //                                     delta *
                                     //                                     (advection_directions[q_point] *
                                     //                                      fe_values.shape_grad(i,q_point))
                                     ) *
                                    rhs_values[q_point] *
                                    fe_values_T.JxW (q_point));
                    //                    std::cout << component_i << std::endl;
                    
                }
            
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary()
                    &&
                    (cell->face(face_number)->boundary_id() == 2))
                {
                    fe_face_values_T.reinit (cell, face_number);
                    for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
                    {
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            cell_rhs(i) += (heatflux_values[q_point] / (phi0*Nu) *
                                            fe_face_values_T.shape_value(i,q_point) *
                                            fe_face_values_T.JxW(q_point));
                    }
                }
            
            cell->get_dof_indices (local_dof_indices);
            
            for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            {
                nontime_rhs_T(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_rock ()
    {
        //        std::cout << "   Solving for rock system..." << std::endl;
        SparseDirectUMFPACK  A_direct;
        A_direct.initialize(system_matrix_rock);
        A_direct.vmult (solution_rock, system_rhs_rock);
        constraints_rock.distribute (solution_rock);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_fluid ()
    {
        //        std::cout << "   Solving for p_f..." << std::endl;
        SparseDirectUMFPACK  pf_direct;
        pf_direct.initialize(system_matrix_pf);
        pf_direct.vmult (solution_pf, system_rhs_pf);
        
        assemble_system_vf ();
        //        std::cout << "   Solving for v_f..." << std::endl;
        SparseDirectUMFPACK  vf_direct;
        vf_direct.initialize(system_matrix_vf);
        vf_direct.vmult (solution_vf, system_rhs_vf);
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
        
        std::map<types::global_dof_index,double> boundary_values;
        ExactSolution_phi<dim> phi_boundary_values;
        phi_boundary_values.set_time(time);
        
        VectorTools::interpolate_boundary_values (dof_handler_phi,
                                                  1,
                                                  phi_boundary_values,
                                                  boundary_values);
        MatrixTools::apply_boundary_values (boundary_values,
                                            system_matrix_phi,
                                            solution_phi,
                                            system_rhs_phi);
        
        //        std::cout << "   Solving for phi..." << std::endl;
        
        SparseDirectUMFPACK  phi_direct;
        phi_direct.initialize(system_matrix_phi);
        phi_direct.vmult (solution_phi, system_rhs_phi);
        
        hanging_node_constraints_phi.distribute (solution_phi);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::solve_T ()
    {
        mass_matrix_T_old.vmult(system_rhs_T, old_solution_T);
        system_rhs_T.add(timestep,nontime_rhs_T);
        
        system_matrix_T.copy_from(mass_matrix_T);
        system_matrix_T.add(timestep,nontime_matrix_T);
        
        std::map<types::global_dof_index,double> boundary_values;
        ExactSolution_T<dim> diri_values;
        diri_values.set_time(time);
        VectorTools::interpolate_boundary_values (dof_handler_T,
                                                  1,
                                                  diri_values,
                                                  boundary_values);
        
        hanging_node_constraints_T.condense (system_rhs_T);
        hanging_node_constraints_T.condense (system_matrix_T);
        
        MatrixTools::apply_boundary_values (boundary_values,
                                            system_matrix_T,
                                            solution_T,
                                            system_rhs_T);
        
        //        std::cout << "   Solving for T..." << std::endl;
        
        SparseDirectUMFPACK  T_direct;
        T_direct.initialize(system_matrix_T);
        T_direct.vmult (solution_T, system_rhs_T);
        
        hanging_node_constraints_T.distribute (solution_T);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::output_results ()  const
    {
        // ROCK
        std::vector<std::string> solution_names (dim, "velocity");
        solution_names.push_back ("pressure");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
        DataOut<dim> data_out_rock;
        data_out_rock.attach_dof_handler (dof_handler_rock);
        data_out_rock.add_data_vector (solution_rock, solution_names,
                                       DataOut<dim>::type_dof_data,
                                       data_component_interpretation);
        data_out_rock.build_patches ();
        std::ostringstream filename_rock;
        filename_rock << "solution_rock"+ Utilities::int_to_string(timestep_number, 3) + ".vtk";
        std::ofstream output_rock (filename_rock.str().c_str());
        data_out_rock.write_vtk (output_rock);
        
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
        data_out_fluid.add_data_vector (dof_handler_vf, solution_vf,
                                        "v_f");
        
        data_out_fluid.build_patches ();
        const std::string filename_fluid = "solution_fluid-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output_fluid (filename_fluid.c_str());
        data_out_fluid.write_vtk (output_fluid);
        
        //PHI
        DataOut<dim> data_out_phi;
        
        data_out_phi.attach_dof_handler(dof_handler_phi);
        data_out_phi.add_data_vector(solution_phi, "phi");
        data_out_phi.build_patches();
        
        const std::string filename_phi = "solution_phi-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output_phi(filename_phi.c_str());
        data_out_phi.write_vtk(output_phi);
        
        //T
        DataOut<dim> data_out_T;
        
        data_out_T.attach_dof_handler(dof_handler_T);
        data_out_T.add_data_vector(solution_T, "T");
        
        data_out_T.build_patches();
        
        const std::string T_filename = "T_solution-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output_T(T_filename.c_str());
        data_out_T.write_vtk(output_T);
        
    }
    
    template <int dim>
    void FullMovingMesh<dim>::move_mesh ()
    {
        std::cout << "   Moving mesh..." << std::endl;
        
        std::vector<bool> vertex_touched (triangulation.n_vertices(),
                                          false);
        
        for (typename DoFHandler<dim>::active_cell_iterator
             cell = dof_handler_rock.begin_active ();
             cell != dof_handler_rock.end(); ++cell)
            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                if (vertex_touched[cell->vertex_index(v)] == false)
                {
                    vertex_touched[cell->vertex_index(v)] = true;
                    Point<dim> vertex_displacement;
                    for (unsigned int d=0; d<dim; ++d)
                    {vertex_displacement[d]
                        = solution_rock.block(0)(cell->vertex_dof_index(v,d));
                    }
                    cell->vertex(v) += vertex_displacement*timestep;
                    //                    std::cout << cell->diameter() << std::endl;
                }
    }
    
    template <int dim>
    void FullMovingMesh<dim>::print_mesh ()
    {
        std::ofstream out ("grid-"
                           + Utilities::int_to_string(timestep_number, 3) +
                           ".eps");
        GridOut grid_out;
        grid_out.write_eps (triangulation, out);
    }
    
    template <int dim>
    void FullMovingMesh<dim>::compute_errors ()
    {
        
        const ComponentSelectFunction<dim> pressure_mask (dim, dim+1);
        const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim), dim+1);
        QTrapez<1>     q_trapez;
        QIterated<dim> quadrature (q_trapez, degree_rock+2);
        
        // ROCK ERRORS
        ExactSolution_rock<dim> exact_solution_rock;
        Vector<double> cellwise_errors_rock (triangulation.n_active_cells());
        
        VectorTools::integrate_difference (dof_handler_rock, solution_rock, exact_solution_rock,
                                           cellwise_errors_rock, quadrature,
                                           VectorTools::L2_norm,
                                           &pressure_mask);
        const double pr_l2_error = cellwise_errors_rock.l2_norm();
        
        VectorTools::integrate_difference (dof_handler_rock, solution_rock, exact_solution_rock,
                                           cellwise_errors_rock, quadrature,
                                           VectorTools::L2_norm,
                                           &velocity_mask);
        const double vr_l2_error = cellwise_errors_rock.l2_norm();
        
        std::cout << "   Errors: ||e_pr||_L2  = " << pr_l2_error
        << "  " << std::endl << "           ||e_vr||_L2  = " << vr_l2_error
        << std::endl;
        
        // U AND PF ERRORS
        ExactSolution_pf<dim> exact_solution_pf;
        Vector<double> cellwise_errors_pf (triangulation.n_active_cells());
        
        VectorTools::integrate_difference (dof_handler_pf, solution_pf, exact_solution_pf,
                                           cellwise_errors_pf, quadrature,
                                           VectorTools::L2_norm,
                                           &velocity_mask);
        const double u_l2_error = cellwise_errors_pf.l2_norm();

        VectorTools::integrate_difference (dof_handler_pf, solution_pf, exact_solution_pf, cellwise_errors_pf, quadrature,
                                           VectorTools::L2_norm,
                                           &pressure_mask);
        const double pf_l2_error = cellwise_errors_pf.l2_norm();
        
        std::cout << "           ||e_pf||_L2  = " << pf_l2_error
        << "  " << std::endl << "           ||e_u||_L2   = " << u_l2_error
        << std::endl;
        
        // VF ERRORS
        ExactSolution_vf<dim> exact_solution_vf;
        
        Vector<float> difference_per_cell (triangulation.n_active_cells());
        VectorTools::integrate_difference (dof_handler_vf,
                                           solution_vf,
                                           exact_solution_vf,
                                           difference_per_cell,
                                           QGauss<dim>(3),
                                           VectorTools::L2_norm);
        const double vf_l2_error = difference_per_cell.l2_norm();
        std::cout << "           ||e_vf||_L2  = " << vf_l2_error
        << ",  " << std::endl;
        
        ExactSolution_phi<dim> exact_solution_phi;
        exact_solution_phi.set_time(time);
        
        Vector<double> cellwise_errors_phi (triangulation.n_active_cells());
        
        VectorTools::integrate_difference (dof_handler_phi, solution_phi, exact_solution_phi,cellwise_errors_phi, quadrature,
                                           VectorTools::L2_norm);
        
        const double phi_L2_error = cellwise_errors_phi.l2_norm();
        
        std::cout << "           ||e_phi||_L2 = " << phi_L2_error << std::endl;
        
        ExactSolution_T<dim> exact_solution_T;
        exact_solution_T.set_time(time);
        
        Vector<double> cellwise_errors_T (triangulation.n_active_cells());
        
        VectorTools::integrate_difference (dof_handler_T, solution_T, exact_solution_T, cellwise_errors_T, quadrature,
                                           VectorTools::L2_norm);
        
        const double T_L2_error = cellwise_errors_T.l2_norm();
        
        std::cout << "           ||e_T||_L2   = " << T_L2_error << std::endl;
        }
    
    template <int dim>
    void FullMovingMesh<dim>::run ()
    {
        timestep_number = 0;
        time = 0.0;

        std::cout << "   Problem degrees: " << degree_rock << ", " << degree_pf << ", " << degree_vf << ", " << degree_phi << ", " << degree_T << "." << std::endl;

        make_initial_grid ();
        print_mesh ();
        setup_dofs_rock ();
        setup_dofs_fluid ();
        setup_dofs_phi ();
        setup_dofs_T ();

        VectorTools::interpolate(dof_handler_vf, InitialFunction_vf<dim>(), solution_vf);
        VectorTools::interpolate(dof_handler_phi, InitialFunction_phi<dim>(), old_solution_phi);
        solution_phi = old_solution_phi;
        VectorTools::interpolate(dof_handler_T, InitialFunction_T<dim>(), old_solution_T);
        solution_T = old_solution_T;

        std::cout << "===========================================" << std::endl;

        apply_BC_rock ();
        std::cout << "   Assembling at timestep number " << timestep_number << " for rock and fluid..." <<  std::endl << std::flush;
        assemble_system_rock ();
        solve_rock ();

        assemble_system_pf ();
        solve_fluid ();
        
        output_results ();
        compute_errors ();
        timestep = timestep_size;

        while (timestep_number < total_timesteps)
        {
            std::cout << "   Moving to next time step" << std::endl
            << "===========================================" << std::endl;

            time += timestep;
            ++timestep_number;

            std::cout << "   Assembling at timestep number " << timestep_number << " from phi..." <<  std::endl << std::flush;

//            Timer timer;
//            timer.start ();

            assemble_system_phi ();
            assemble_rhs_phi ();
            solve_phi (); // now have new solution and old solution

            assemble_system_T ();  //needs both phi solution and old phi solution
            assemble_rhs_T ();
            solve_T ();

            apply_BC_rock ();
            assemble_system_rock ();
            solve_rock ();

            assemble_system_pf ();
            solve_fluid ();

//            timer.stop ();
//            std::cout << "   Elapsed CPU time: " << timer() << " seconds." << std::endl;
//            timer.reset ();

            output_results ();
            //            if (timestep_number == error_timestep)
            compute_errors ();

            old_solution_phi = solution_phi;
            old_solution_T = solution_T;

            //            print_mesh ();
            //            move_mesh ();
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
        FullMovingMesh<2> flow_problem(degree_rock, degree_pf, degree_vf, degree_phi, degree_T);
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
